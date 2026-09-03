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

"""NRLLanceDB: LangChain VectorStore wrapper with NRL-aware metadata parsing.

NRL (NemoRetriever Library) stores the ``metadata`` column as ``str(meta)`` —
a Python repr string using single quotes and Python boolean literals
(``True``/``False``/``None``) — NOT valid JSON.  The standard LangChain
LanceDB VectorStore's ``results_to_docs`` passes this raw string directly to
``Document(metadata=...)`` which fails Pydantic validation (expects a dict).

This module provides ``NRLLanceDB``, a subclass of
``langchain_community.vectorstores.LanceDB`` that overrides ``results_to_docs``
to correctly handle NRL's storage format and expose every LanceDB column
(except the raw float ``vector``) in the returned Document metadata.

Usage::

    from nvidia_rag.utils.vdb.lancedb.nrl_lancedb import NRLLanceDB

    store = NRLLanceDB(
        connection=lancedb_connection,
        embedding=embedding_model,
        vector_key="vector",
        id_key="source_id",
        text_key="text",
        table_name="my_collection",
    )
    docs = store.similarity_search("query", k=5)

NOTE: Do NOT add ``import lancedb`` to this module.  LanceDB is not
fork-safe; ``lancedb.connect()`` must remain inside the caller's method.
The ``langchain_community.vectorstores.LanceDB`` base class uses lazy
imports internally and is safe to reference at module level here.
"""

import ast
import logging
from typing import Any

from langchain_community.vectorstores import LanceDB as LangchainLanceDB
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Columns intentionally excluded from Document.metadata:
#   - "vector":  High-dimensional float array — large and not useful for consumers.
#   - "text":    Carried as Document.page_content to avoid duplication.
_SKIP_IN_METADATA: frozenset[str] = frozenset({"vector", "text"})


def _parse_nrl_metadata(val: Any) -> dict:
    """Parse NRL's ``str(dict)`` metadata representation to a Python dict.

    NRL's ``_build_lancedb_rows_from_df`` stores the metadata field as::

        "metadata": str(meta)

    Example stored value::

        "{'has_text': True, 'needs_ocr_for_text': False, 'dpi': 200,
          'source_path': '/tmp-data/.../file.pdf', 'error': None}"

    ``ast.literal_eval`` handles Python literals (``True``, ``False``,
    ``None``) that are not valid JSON tokens, making it the correct parser.
    Returns an empty dict for null or unparseable values instead of raising.
    """
    if not val:
        return {}
    if isinstance(val, dict):
        # Already a dict — some alternative code path may materialise it.
        return val
    try:
        parsed = ast.literal_eval(str(val))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        logger.debug("_parse_nrl_metadata: could not parse value: %r", val)
        return {}


class NRLLanceDB(LangchainLanceDB):
    """LangChain LanceDB VectorStore with NRL-aware metadata handling.

    Overrides ``results_to_docs`` to correctly handle the document format
    produced by NRL (NemoRetriever Library) ingestion.

    All columns except ``vector`` and ``text`` are preserved in
    ``Document.metadata`` so downstream code (reranker, citations,
    confidence filtering) has full provenance.
    """

    def results_to_docs(self, results: Any, score: bool = False) -> list:
        """Convert LanceDB ANN results to a list of LangChain Documents.

        Every column except ``vector`` and ``text`` is surfaced in
        ``Document.metadata``.  The ``metadata`` column (NRL Python repr
        string) is parsed with ``ast.literal_eval`` and stored back under
        the same ``"metadata"`` key as a proper Python dict.

        Parameters
        ----------
        results:
            PyArrow RecordBatch / Table returned by a LanceDB vector search.
        score:
            When ``True``, return ``(Document, float)`` tuples where the
            float comes from the ``_distance`` or ``_relevance_score`` column.
            When ``False``, return plain ``Document`` objects.

        Returns
        -------
        list[Document] | list[tuple[Document, float]]
        """
        columns = results.schema.names

        # Identify the score column.
        # LanceDB ANN search produces _distance; rerankers may use _relevance_score.
        if "_distance" in columns:
            score_col = "_distance"
        elif "_relevance_score" in columns:
            score_col = "_relevance_score"
        else:
            score_col = None

        # self._text_key is set to "text" by get_langchain_vectorstore.
        text_key = self._text_key

        docs = []
        for idx in range(len(results)):
            # ── Text content → page_content ──────────────────────────────
            text = (
                results[text_key][idx].as_py() if text_key in columns else ""
            ) or ""

            # ── Collect every non-vector, non-text column into metadata ───
            # The loop is intentionally generic so it stays correct even when
            # NRL adds or renames columns in the future.
            metadata: dict[str, Any] = {}
            for col in columns:
                if col in _SKIP_IN_METADATA:
                    continue
                try:
                    metadata[col] = results[col][idx].as_py()
                except Exception:
                    # Defensive: skip columns that fail to deserialise.
                    logger.debug(
                        "NRLLanceDB.results_to_docs: skipping column '%s' at idx %d"
                        " — as_py() raised an exception",
                        col,
                        idx,
                    )

            # ── Parse the NRL metadata column ─────────────────────────────
            # NRL stores the metadata dict as a Python repr string in the
            # "metadata" column (str(meta)).  Pop the raw string, parse it
            # with ast.literal_eval, and store the resulting dict back under
            # the same "metadata" key so downstream code always receives a dict.
            raw_nrl_meta = metadata.pop("metadata", None)
            metadata["metadata"] = _parse_nrl_metadata(raw_nrl_meta)

            doc = Document(page_content=text, metadata=metadata)

            if score and score_col is not None:
                docs.append((doc, results[score_col][idx].as_py()))
            else:
                docs.append(doc)

        return docs

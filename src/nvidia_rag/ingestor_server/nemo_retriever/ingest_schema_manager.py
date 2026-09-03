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

"""IngestSchemaManager: schema isolation layer over the raw NRL result DataFrame.

When NRL renames a column, only the class constants in this file change.
All other code uses the stable method API.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from nemo_retriever.vector_store.lancedb_utils import build_lancedb_rows


class IngestSchemaManager:
    """Wraps the raw NRL result DataFrame and provides a stable API
    regardless of how NRL's internal column names evolve.

    NRL canonical column names are defined as class constants here.
    Update only these constants when NRL changes its schema.
    """

    # --- Column name constants (update here only when NRL changes schema) ---
    # Source path is in two locations; build_lancedb_row prefers metadata.source_path.
    COL_PATH = "path"
    COL_TEXT = "text"
    COL_PAGE_NUMBER = "page_number"
    COL_METADATA = "metadata"  # dict: source_path, content_metadata, embedding
    COL_EMBEDDING = "text_embeddings_1b_v2"  # matches EmbedParams.output_column default

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    # --- Document-level accessors ---

    def source_ids(self) -> list[str]:
        """Unique document identifiers (paths) present in the result."""
        if self.COL_PATH not in self._df.columns:
            return []
        return [
            p
            for p in self._df[self.COL_PATH].dropna().unique()
            if isinstance(p, str) and p.strip()
        ]

    def succeeded_sources(self) -> list[str]:
        """Sources that produced at least one embedded chunk."""
        if (
            self.COL_EMBEDDING not in self._df.columns
            or self.COL_PATH not in self._df.columns
        ):
            return []

        def _has_embedding(v: Any) -> bool:
            if not isinstance(v, dict):
                return False
            emb = v.get("embedding")
            return isinstance(emb, list) and len(emb) > 0

        mask = self._df[self.COL_EMBEDDING].apply(_has_embedding)
        return [
            p
            for p in self._df.loc[mask, self.COL_PATH].dropna().unique()
            if isinstance(p, str) and p.strip()
        ]

    def failed_sources(self) -> list[str]:
        """Sources that produced only error rows (no embedded chunk)."""
        all_ids = set(self.source_ids())
        succeeded = set(self.succeeded_sources())
        return list(all_ids - succeeded)

    def row_count(self) -> int:
        """Total number of rows in the result DataFrame."""
        return len(self._df)

    # --- VDB write path ---

    def to_canonical_records(self) -> list[dict]:
        """Convert DataFrame to NRL canonical VDB records.

        Delegates entirely to NRL's own builder — no custom field mapping here.
        Output shape per record:
            vector, text, metadata (JSON), source (JSON), page_number,
            pdf_page, pdf_basename, filename, source_id, path.

        NOTE: When NRL PR #1822 merges, replace ``build_lancedb_rows`` with:
            from nemo_retriever.vector_store.vdb_records import build_vdb_records
            return build_vdb_records(self._df)
        """
        if self._df.empty:
            return []
        return build_lancedb_rows(self._df)

    def to_raw_records(self) -> list[dict]:
        """Convert DataFrame to raw NRL records for VDB ingestion.

        Returns the NRL result DataFrame as a flat ``list[dict]`` via
        ``DataFrame.to_dict("records")``.  This is the format expected by
        any VDB backend's ``write_to_index()`` that consumes raw NRL rows
        (currently LanceDB; Elasticsearch and Milvus backends will use the
        same method when NRL support is added to them).

        Usage::

            schema_mgr = IngestSchemaManager(df)
            raw_records = schema_mgr.to_raw_records()
            vdb_op.write_to_index(raw_records)

        Returns
        -------
        list[dict]
            One dict per DataFrame row; empty list when the DataFrame is empty.
        """
        if self._df.empty:
            return []
        return self._df.to_dict("records")

    # --- Summarisation adapter ---

    def to_nv_ingest_results_format(self) -> list[list[dict]]:
        """Convert DataFrame to the ``list[list[dict]]`` shape that
        ``generate_document_summaries()`` currently expects.

        Each inner list represents one source document.  Each element carries:

        * ``document_type``                          — ``"text"`` for NRL text chunks
        * ``metadata.source_metadata.source_id``    — full file path
        * ``metadata.content_metadata.page_number`` — page number
        * ``metadata.content``                       — text content for summarisation

        This is the only method that knows about the OLD NV-Ingest result shape.
        Remove once ``summarisation.py`` is refactored to accept canonical records.
        """
        if self._df.empty:
            return []

        if self.COL_PATH not in self._df.columns:
            return []

        has_page = self.COL_PAGE_NUMBER in self._df.columns
        has_text = self.COL_TEXT in self._df.columns

        # Drop rows with missing or blank paths before grouping.
        clean_df = self._df.dropna(subset=[self.COL_PATH])
        clean_df = clean_df[clean_df[self.COL_PATH].astype(str).str.strip() != ""]

        result: list[list[dict]] = []
        for source_id, group_df in clean_df.groupby(self.COL_PATH, sort=False):
            doc_records: list[dict] = []
            for _, row in group_df.iterrows():
                try:
                    page_number = int(row.get(self.COL_PAGE_NUMBER)) if has_page else 1
                except (TypeError, ValueError):
                    page_number = 1

                text_content = (
                    str(row.get(self.COL_TEXT))
                    if has_text and isinstance(row.get(self.COL_TEXT), str)
                    else ""
                )

                doc_records.append(
                    {
                        "document_type": "text",
                        "metadata": {
                            "source_metadata": {"source_id": str(source_id)},
                            "content_metadata": {"page_number": page_number},
                            "content": text_content,
                        },
                    }
                )

            if doc_records:
                result.append(doc_records)

        return result

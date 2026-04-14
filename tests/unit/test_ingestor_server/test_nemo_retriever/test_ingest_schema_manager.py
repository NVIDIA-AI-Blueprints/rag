# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for IngestSchemaManager."""

from unittest.mock import patch

import pandas as pd

from nvidia_rag.ingestor_server.nemo_retriever.ingest_schema_manager import (
    IngestSchemaManager,
)


class TestIngestSchemaManagerSourceIds:
    def test_empty_when_no_path_column(self) -> None:
        df = pd.DataFrame({"text": ["a"]})
        mgr = IngestSchemaManager(df)
        assert mgr.source_ids() == []

    def test_unique_non_empty_paths(self) -> None:
        df = pd.DataFrame(
            {
                "path": ["/a/x.pdf", "/a/x.pdf", "", None, "/b/y.pdf"],
            }
        )
        mgr = IngestSchemaManager(df)
        assert set(mgr.source_ids()) == {"/a/x.pdf", "/b/y.pdf"}


class TestIngestSchemaManagerSucceededFailed:
    def test_succeeded_requires_embedding_column_and_dict(self) -> None:
        df = pd.DataFrame(
            {
                "path": ["/doc.pdf"],
                "text_embeddings_1b_v2": [{"embedding": [0.1, 0.2]}],
            }
        )
        mgr = IngestSchemaManager(df)
        assert mgr.succeeded_sources() == ["/doc.pdf"]

    def test_succeeded_empty_when_embedding_missing_or_empty(self) -> None:
        df = pd.DataFrame(
            {
                "path": ["/doc.pdf", "/doc2.pdf"],
                "text_embeddings_1b_v2": [{"embedding": []}, {"not_embedding": True}],
            }
        )
        mgr = IngestSchemaManager(df)
        assert mgr.succeeded_sources() == []

    def test_failed_sources_is_difference(self) -> None:
        df = pd.DataFrame(
            {
                "path": ["/ok.pdf", "/ok.pdf", "/bad.pdf"],
                "text_embeddings_1b_v2": [
                    {"embedding": [0.1]},
                    {"embedding": [0.2]},
                    {"embedding": []},
                ],
            }
        )
        mgr = IngestSchemaManager(df)
        assert set(mgr.succeeded_sources()) == {"/ok.pdf"}
        assert set(mgr.failed_sources()) == {"/bad.pdf"}


class TestIngestSchemaManagerRecords:
    def test_row_count(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        assert IngestSchemaManager(df).row_count() == 3

    def test_to_raw_records_empty(self) -> None:
        df = pd.DataFrame()
        assert IngestSchemaManager(df).to_raw_records() == []

    def test_to_raw_records_roundtrip(self) -> None:
        df = pd.DataFrame({"path": ["/x"], "text": ["hello"]})
        raw = IngestSchemaManager(df).to_raw_records()
        assert raw == [{"path": "/x", "text": "hello"}]

    @patch(
        "nvidia_rag.ingestor_server.nemo_retriever.ingest_schema_manager.build_lancedb_rows"
    )
    def test_to_canonical_records_delegates(self, mock_build: object) -> None:
        mock_build.return_value = [{"vector": [1.0]}]
        df = pd.DataFrame({"path": ["/p"]})
        mgr = IngestSchemaManager(df)
        assert mgr.to_canonical_records() == [{"vector": [1.0]}]
        mock_build.assert_called_once_with(df)

    @patch(
        "nvidia_rag.ingestor_server.nemo_retriever.ingest_schema_manager.build_lancedb_rows"
    )
    def test_to_canonical_records_empty_skips_builder(self, mock_build: object) -> None:
        df = pd.DataFrame()
        assert IngestSchemaManager(df).to_canonical_records() == []
        mock_build.assert_not_called()


class TestToNvIngestResultsFormat:
    def test_empty_dataframe(self) -> None:
        assert IngestSchemaManager(pd.DataFrame()).to_nv_ingest_results_format() == []

    def test_missing_path_column(self) -> None:
        df = pd.DataFrame({"text": ["x"]})
        assert IngestSchemaManager(df).to_nv_ingest_results_format() == []

    def test_groups_by_path_with_page_and_text(self) -> None:
        df = pd.DataFrame(
            {
                "path": ["/a.pdf", "/a.pdf", "/b.pdf"],
                "page_number": [1, 2, 1],
                "text": ["hello", "world", "solo"],
            }
        )
        mgr = IngestSchemaManager(df)
        out = mgr.to_nv_ingest_results_format()
        assert len(out) == 2
        by_source = {
            group[0]["metadata"]["source_metadata"]["source_id"]: group for group in out
        }
        assert len(by_source["/a.pdf"]) == 2
        assert (
            by_source["/a.pdf"][0]["metadata"]["content_metadata"]["page_number"] == 1
        )
        assert (
            by_source["/a.pdf"][1]["metadata"]["content_metadata"]["page_number"] == 2
        )
        assert by_source["/b.pdf"][0]["metadata"]["content"] == "solo"

    def test_invalid_page_number_defaults_to_one(self) -> None:
        df = pd.DataFrame(
            {
                "path": ["/x.pdf"],
                "page_number": ["not-int"],
                "text": ["t"],
            }
        )
        mgr = IngestSchemaManager(df)
        row = mgr.to_nv_ingest_results_format()[0][0]
        assert row["metadata"]["content_metadata"]["page_number"] == 1

    def test_non_string_text_becomes_empty(self) -> None:
        df = pd.DataFrame(
            {
                "path": ["/x.pdf"],
                "page_number": [1],
                "text": [123],
            }
        )
        mgr = IngestSchemaManager(df)
        row = mgr.to_nv_ingest_results_format()[0][0]
        assert row["metadata"]["content"] == ""

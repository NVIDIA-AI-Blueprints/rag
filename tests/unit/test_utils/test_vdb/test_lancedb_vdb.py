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

"""Unit tests for LanceDBVDB and helpers in lancedb_vdb module."""

import asyncio
import builtins
import importlib
import sys
import unittest
from concurrent.futures import Future
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pyarrow as pa
import pytest
import requests
from langchain_core.documents import Document

from nvidia_rag.rag_server.response_generator import APIError, ErrorCodeMapping
from nvidia_rag.utils.health_models import ServiceStatus
from nvidia_rag.utils.vdb import (
    DEFAULT_DOCUMENT_INFO_COLLECTION,
    DEFAULT_METADATA_SCHEMA_COLLECTION,
)
from nvidia_rag.utils.vdb.lancedb.lancedb_vdb import (
    _LANCEDB_INSTALL_MSG,
    LanceDBVDB,
    _import_lancedb,
    _parse_nrl_metadata,
)


class TestParseNrlMetadata(unittest.TestCase):
    """Tests for _parse_nrl_metadata."""

    def test_empty_returns_empty_dict(self) -> None:
        assert _parse_nrl_metadata(None) == {}
        assert _parse_nrl_metadata("") == {}
        assert _parse_nrl_metadata(0) == {}

    def test_dict_passthrough(self) -> None:
        d = {"page_number": 1, "foo": "bar"}
        assert _parse_nrl_metadata(d) is d

    def test_literal_eval_string(self) -> None:
        s = "{'page_number': 2, 'x': True}"
        out = _parse_nrl_metadata(s)
        assert out == {"page_number": 2, "x": True}

    def test_invalid_returns_empty(self) -> None:
        assert _parse_nrl_metadata("not a dict {{{") == {}
        assert _parse_nrl_metadata("[1, 2]") == {}


class TestImportLancedb(unittest.TestCase):
    """Tests for _import_lancedb."""

    def test_raises_clear_message_when_import_fails(self) -> None:
        real_import = builtins.__import__

        def selective_import(
            name: str, globals_=None, locals_=None, fromlist=(), level: int = 0
        ):
            if level == 0 and name == "lancedb":
                raise ImportError("simulated missing lancedb")
            return real_import(name, globals_, locals_, fromlist, level)

        mod = importlib.import_module("nvidia_rag.utils.vdb.lancedb.lancedb_vdb")
        saved = sys.modules.pop("lancedb", None)
        try:
            with patch.object(builtins, "__import__", side_effect=selective_import):
                with pytest.raises(ImportError) as exc_info:
                    mod._import_lancedb()
                assert _LANCEDB_INSTALL_MSG in str(exc_info.value)
        finally:
            if saved is not None:
                sys.modules["lancedb"] = saved
            elif "lancedb" not in sys.modules:
                importlib.import_module("lancedb")

    def test_returns_module_when_available(self) -> None:
        mod = _import_lancedb()
        assert mod is not None
        assert hasattr(mod, "connect")


class TestLanceDBVDBInit(unittest.TestCase):
    """Basic LanceDBVDB construction and properties."""

    def test_init_defaults(self) -> None:
        vdb = LanceDBVDB("tbl", "/data/lancedb")
        assert vdb.collection_name == "tbl"
        assert vdb.uri == "/data/lancedb"
        assert vdb.hybrid is False
        assert vdb.overwrite is False
        assert vdb._metadata_schema_collection_initialized is False
        assert vdb._document_info_collection_initialized is False

    def test_collection_name_setter(self) -> None:
        vdb = LanceDBVDB("a", "/x")
        vdb.collection_name = "b"
        assert vdb._table_name == "b"
        assert vdb.collection_name == "b"


class TestCheckHealth(unittest.TestCase):
    """Tests for LanceDBVDB.check_health."""

    def test_no_uri_skipped(self) -> None:
        vdb = LanceDBVDB("t", "")
        out = asyncio.run(vdb.check_health())
        assert out["status"] == ServiceStatus.SKIPPED.value
        assert "No URI" in (out.get("error") or "")

    @patch("nvidia_rag.utils.vdb.lancedb.lancedb_vdb._import_lancedb")
    def test_healthy(self, mock_imp: MagicMock) -> None:
        mock_db = MagicMock()
        mock_db.table_names.return_value = ["a", "b"]
        mock_imp.return_value.connect.return_value = mock_db
        vdb = LanceDBVDB("t", "/tmp/lance")
        out = asyncio.run(vdb.check_health())
        assert out["status"] == ServiceStatus.HEALTHY.value
        assert out["tables"] == 2

    @patch("nvidia_rag.utils.vdb.lancedb.lancedb_vdb._import_lancedb")
    def test_error_on_connect(self, mock_imp: MagicMock) -> None:
        mock_imp.return_value.connect.side_effect = RuntimeError("boom")
        vdb = LanceDBVDB("t", "/tmp/lance")
        out = asyncio.run(vdb.check_health())
        assert out["status"] == ServiceStatus.ERROR.value
        assert "boom" in (out.get("error") or "")


class TestIngestInterface(unittest.TestCase):
    """create_index, write_to_index, run, run_async, reindex, retrieval stub."""

    @patch("nvidia_rag.utils.vdb.lancedb.lancedb_vdb.Path.mkdir")
    @patch("nvidia_rag.utils.vdb.lancedb.lancedb_vdb._import_lancedb")
    @patch(
        "nemo_retriever.vector_store.lancedb_utils.lancedb_schema",
    )
    def test_create_index_creates_when_missing(
        self,
        mock_lancedb_schema: MagicMock,
        mock_imp: MagicMock,
        _mkdir: MagicMock,
    ) -> None:
        dim = 4
        schema = pa.schema(
            [
                pa.field("text", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), dim)),
            ]
        )
        mock_lancedb_schema.return_value = schema

        mock_db = MagicMock()
        mock_db.open_table.side_effect = Exception("no table")
        mock_imp.return_value.connect.return_value = mock_db

        mock_cfg = Mock()
        mock_cfg.embeddings.dimensions = dim
        vdb = LanceDBVDB("mytbl", "/tmp/db", config=mock_cfg)
        vdb.create_index()

        mock_db.create_table.assert_called_once()
        call_kw = mock_db.create_table.call_args
        assert call_kw[0][0] == "mytbl"
        assert call_kw[1].get("mode") == "create"

    @patch("nvidia_rag.utils.vdb.lancedb.lancedb_vdb.logger")
    def test_write_to_index_empty_records(self, mock_logger: MagicMock) -> None:
        vdb = LanceDBVDB("t", "/x")
        vdb.write_to_index([])
        mock_logger.warning.assert_called()

    @patch(
        "nemo_retriever.vector_store.lancedb_store.handle_lancedb",
    )
    def test_write_to_index_overwrite_delegates_to_handle_lancedb(
        self,
        mock_handle: MagicMock,
    ) -> None:
        vdb = LanceDBVDB("t", "/x", overwrite=True, hybrid=True)
        vdb.write_to_index([{"row": 1}])
        mock_handle.assert_called_once_with([{"row": 1}], "/x", "t", hybrid=True)

    @patch("nvidia_rag.utils.vdb.lancedb.lancedb_vdb._import_lancedb")
    @patch(
        "nemo_retriever.vector_store.lancedb_store._build_lancedb_rows_from_df",
        return_value=[],
    )
    def test_write_to_index_append_no_rows_after_transform(
        self,
        _build: MagicMock,
        _imp: MagicMock,
    ) -> None:
        vdb = LanceDBVDB("t", "/x", overwrite=False)
        vdb.write_to_index([{"x": 1}])
        _imp.return_value.connect.assert_not_called()

    def test_run_calls_create_and_write(self) -> None:
        vdb = LanceDBVDB("t", "/x")
        with (
            patch.object(vdb, "create_index") as mock_ci,
            patch.object(vdb, "write_to_index") as mock_w,
        ):
            vdb.run([1, 2])
        mock_ci.assert_called_once()
        mock_w.assert_called_once_with([1, 2])

    def test_run_async_resolves_future(self) -> None:
        vdb = LanceDBVDB("t", "/x")
        fut: Future = Future()
        fut.set_result([{"a": 1}])
        with (
            patch.object(vdb, "create_index"),
            patch.object(vdb, "write_to_index") as mock_w,
        ):
            out = vdb.run_async(fut)
        assert out == [{"a": 1}]
        mock_w.assert_called_once_with([{"a": 1}])

    def test_reindex_temporarily_sets_overwrite(self) -> None:
        vdb = LanceDBVDB("t", "/x", overwrite=False)
        overwrite_during_run: list[bool] = []

        def capture_run(_records: list) -> None:
            overwrite_during_run.append(vdb.overwrite)

        with patch.object(vdb, "run", side_effect=capture_run):
            vdb.reindex([1])
        assert overwrite_during_run == [True]
        assert vdb.overwrite is False

    def test_retrieval_not_implemented(self) -> None:
        vdb = LanceDBVDB("t", "/x")
        with pytest.raises(NotImplementedError, match="retrieval_langchain"):
            vdb.retrieval([])


class TestCollectionAndDocumentOps(unittest.TestCase):
    """Collection CRUD and document listing."""

    @patch("nvidia_rag.utils.vdb.lancedb.lancedb_vdb.Path.mkdir")
    @patch("nvidia_rag.utils.vdb.lancedb.lancedb_vdb._import_lancedb")
    @patch("nemo_retriever.vector_store.lancedb_utils.lancedb_schema")
    def test_create_collection_idempotent_when_exists(
        self,
        mock_lancedb_schema: MagicMock,
        mock_imp: MagicMock,
        _mkdir: MagicMock,
    ) -> None:
        schema = pa.schema([pa.field("text", pa.string())])
        mock_lancedb_schema.return_value = schema

        mock_db = MagicMock()
        mock_db.open_table.return_value = MagicMock()
        mock_imp.return_value.connect.return_value = mock_db

        vdb = LanceDBVDB("ignored", "/tmp/db")
        vdb.create_collection("exists", dimension=8)
        mock_db.create_table.assert_not_called()

    @patch("nvidia_rag.utils.vdb.lancedb.lancedb_vdb._import_lancedb")
    def test_check_collection_exists(self, mock_imp: MagicMock) -> None:
        mock_db = MagicMock()
        mock_db.table_names.return_value = ["a", "b"]
        mock_imp.return_value.connect.return_value = mock_db
        vdb = LanceDBVDB("t", "/x")
        assert vdb.check_collection_exists("b") is True
        assert vdb.check_collection_exists("c") is False

    @patch("nvidia_rag.utils.vdb.lancedb.lancedb_vdb._import_lancedb")
    def test_delete_collections(self, mock_imp: MagicMock) -> None:
        mock_db = MagicMock()
        mock_db.table_names.return_value = ["c1", "c2"]
        mock_imp.return_value.connect.return_value = mock_db
        vdb = LanceDBVDB("t", "/x")
        with patch.object(vdb, "_delete_from_system_table") as mock_del_sys:
            out = vdb.delete_collections(["c1", "missing"])
        mock_db.drop_table.assert_called_once_with("c1")
        assert "c1" in out["successful"]
        assert out["total_success"] == 1
        assert any(f["collection_name"] == "missing" for f in out["failed"])
        assert mock_del_sys.call_count == 2

    @patch("nvidia_rag.utils.vdb.lancedb.lancedb_vdb._import_lancedb")
    def test_get_documents_with_path_column(self, mock_imp: MagicMock) -> None:
        df = pd.DataFrame(
            {
                "path": ["/docs/a.pdf", "/docs/a.pdf"],
                "metadata": ["{'title': 'T'}", "{'title': 'T'}"],
            }
        )
        mock_table = MagicMock()
        mock_table.to_pandas.return_value = df
        mock_db = MagicMock()
        mock_db.open_table.return_value = mock_table
        mock_imp.return_value.connect.return_value = mock_db

        vdb = LanceDBVDB("t", "/x")
        with (
            patch.object(
                vdb,
                "get_metadata_schema",
                return_value=[{"name": "title"}],
            ),
            patch.object(vdb, "_get_document_info_map", return_value={}),
        ):
            docs = vdb.get_documents("coll")

        assert len(docs) == 1
        assert docs[0]["document_name"] == "a.pdf"
        assert docs[0]["metadata"] == {"title": "T"}

    @patch("nvidia_rag.utils.vdb.lancedb.lancedb_vdb._import_lancedb")
    def test_get_documents_no_source_columns_returns_empty(
        self, mock_imp: MagicMock
    ) -> None:
        mock_table = MagicMock()
        mock_table.to_pandas.return_value = pd.DataFrame({"text": ["x"]})
        mock_db = MagicMock()
        mock_db.open_table.return_value = mock_table
        mock_imp.return_value.connect.return_value = mock_db
        vdb = LanceDBVDB("t", "/x")
        with (
            patch.object(vdb, "get_metadata_schema", return_value=[]),
            patch.object(vdb, "_get_document_info_map", return_value={}),
        ):
            assert vdb.get_documents("c") == []

    @patch("nvidia_rag.utils.vdb.lancedb.lancedb_vdb._import_lancedb")
    def test_delete_documents_updates_result_dict(self, mock_imp: MagicMock) -> None:
        mock_table = MagicMock()
        mock_table.count_rows.side_effect = [10, 8]
        mock_db = MagicMock()
        mock_db.open_table.return_value = mock_table
        mock_imp.return_value.connect.return_value = mock_db

        vdb = LanceDBVDB("t", "/x")
        with patch.object(vdb, "_delete_document_info_entry"):
            rd: dict = {}
            ok = vdb.delete_documents("coll", ["/p/a.pdf"], result_dict=rd)
        assert ok is True
        assert "a.pdf" in rd["deleted"]


class TestRetrieval(unittest.TestCase):
    """retrieval_langchain, get_langchain_vectorstore, filters, image retrieval."""

    def test_add_collection_name_to_retreived_docs(self) -> None:
        docs = [
            Document(page_content="a", metadata={}),
            Document(page_content="b", metadata={"x": 1}),
        ]
        out = LanceDBVDB._add_collection_name_to_retreived_docs(docs, "C")
        assert out[0].metadata["collection_name"] == "C"
        assert out[1].metadata["collection_name"] == "C"

    @patch("nvidia_rag.utils.vdb.lancedb.lancedb_vdb._import_lancedb")
    @patch("nvidia_rag.utils.vdb.lancedb.nrl_lancedb.NRLLanceDB")
    def test_get_langchain_vectorstore(
        self,
        mock_nrl: MagicMock,
        mock_imp: MagicMock,
    ) -> None:
        mock_db = MagicMock()
        mock_imp.return_value.connect.return_value = mock_db
        emb = MagicMock()
        vdb = LanceDBVDB("tbl", "/data", embedding_model=emb)
        vs = vdb.get_langchain_vectorstore("tbl")
        mock_nrl.assert_called_once()
        assert vs is mock_nrl.return_value

    @patch("nvidia_rag.utils.vdb.lancedb.lancedb_vdb._import_lancedb")
    @patch("nvidia_rag.utils.vdb.lancedb.lancedb_vdb.NRLLanceDB", create=True)
    def test_get_langchain_vectorstore_connect_failure(
        self, _nrl: MagicMock, mock_imp: MagicMock
    ) -> None:
        mock_imp.return_value.connect.side_effect = OSError("nope")
        vdb = LanceDBVDB("tbl", "/data")
        with pytest.raises(RuntimeError, match="failed to connect"):
            vdb.get_langchain_vectorstore("tbl")

    def test_retrieval_langchain_success(self) -> None:
        vdb = LanceDBVDB("tbl", "/data", embedding_model=MagicMock())
        doc = Document(page_content="hi", metadata={})
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [doc]
        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = mock_retriever

        with patch.object(
            vdb,
            "_add_collection_name_to_retreived_docs",
            wraps=vdb._add_collection_name_to_retreived_docs,
        ) as wrap_add:
            out = vdb.retrieval_langchain("q", "coll", vectorstore=mock_vs, top_k=3)
        assert len(out) == 1
        assert out[0].metadata.get("collection_name") == "coll"
        wrap_add.assert_called_once()

    def test_retrieval_langchain_connection_error_maps_to_api_error(self) -> None:
        vdb = LanceDBVDB("tbl", "/data", embedding_model=MagicMock())
        mock_retriever = MagicMock()
        mock_retriever.invoke.side_effect = requests.exceptions.ConnectionError()
        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = mock_retriever

        with pytest.raises(APIError) as exc_info:
            vdb.retrieval_langchain("q", "coll", vectorstore=mock_vs)
        assert exc_info.value.status_code == ErrorCodeMapping.SERVICE_UNAVAILABLE

    @patch("nvidia_rag.utils.vdb.lancedb.lancedb_vdb._import_lancedb")
    def test_retrieve_chunks_by_filter_empty_pages(self, _imp: MagicMock) -> None:
        vdb = LanceDBVDB("t", "/x")
        assert vdb.retrieve_chunks_by_filter("c", "src", []) == []

    @patch("nvidia_rag.utils.vdb.lancedb.lancedb_vdb._import_lancedb")
    def test_retrieve_chunks_by_filter_matches_page(self, mock_imp: MagicMock) -> None:
        df = pd.DataFrame(
            {
                "path": ["/doc.pdf"],
                "text": ["chunk"],
                "metadata": ["{'page_number': 3}"],
            }
        )
        mock_table = MagicMock()
        mock_table.to_pandas.return_value = df
        mock_db = MagicMock()
        mock_db.open_table.return_value = mock_table
        mock_imp.return_value.connect.return_value = mock_db

        vdb = LanceDBVDB("t", "/x")
        docs = vdb.retrieve_chunks_by_filter("c", "/doc.pdf", [3], limit=10)
        assert len(docs) == 1
        assert docs[0].page_content == "chunk"

    def test_retrieval_image_langchain_embedding_error_returns_empty(self) -> None:
        vdb = LanceDBVDB("t", "/x")
        emb = MagicMock()
        emb.embed_documents.side_effect = ValueError("bad")
        vdb._embedding_model = emb
        mock_vs = MagicMock()
        out = vdb.retrieval_image_langchain("img", "coll", vectorstore=mock_vs)
        assert out == []

    @patch.object(LanceDBVDB, "retrieve_chunks_by_filter")
    def test_retrieval_image_langchain_delegates_to_filter(
        self,
        mock_filter: MagicMock,
    ) -> None:
        vdb = LanceDBVDB("t", "/x")
        emb = MagicMock()
        emb.embed_documents.return_value = [[0.1, 0.2]]
        vdb._embedding_model = emb

        top_doc = Document(
            page_content="x",
            metadata={
                "path": "/a/b.pdf",
                "metadata": {"page_number": 5},
            },
        )
        mock_vs = MagicMock()
        mock_vs.similarity_search_by_vector_with_relevance_scores.return_value = [
            (top_doc, 0.9)
        ]

        vdb.retrieval_image_langchain("img", "coll", vectorstore=mock_vs, top_k=1)
        mock_filter.assert_called_once()
        call_kw = mock_filter.call_args[1]
        assert call_kw["source_name"] == "/a/b.pdf"
        assert call_kw["page_numbers"] == [5]


class TestMetadataAndCatalog(unittest.TestCase):
    """System tables, catalog helpers."""

    @patch("nvidia_rag.utils.vdb.lancedb.lancedb_vdb._import_lancedb")
    def test_get_metadata_schema_missing_table(self, mock_imp: MagicMock) -> None:
        mock_db = MagicMock()
        mock_db.table_names.return_value = []
        mock_imp.return_value.connect.return_value = mock_db
        vdb = LanceDBVDB("t", "/x")
        assert vdb.get_metadata_schema("any") == []

    @patch("nvidia_rag.utils.vdb.lancedb.lancedb_vdb._import_lancedb")
    def test_get_metadata_schema_reads_json(self, mock_imp: MagicMock) -> None:
        df = pd.DataFrame(
            {
                "collection_name": ["c1"],
                "metadata_schema": ['[{"name": "n"}]'],
            }
        )
        mock_table = MagicMock()
        mock_table.to_pandas.return_value = df
        mock_db = MagicMock()
        mock_db.table_names.return_value = [DEFAULT_METADATA_SCHEMA_COLLECTION]
        mock_db.open_table.return_value = mock_table
        mock_imp.return_value.connect.return_value = mock_db

        vdb = LanceDBVDB("t", "/x")
        assert vdb.get_metadata_schema("c1") == [{"name": "n"}]

    @patch("nvidia_rag.utils.vdb.lancedb.lancedb_vdb._import_lancedb")
    def test_get_document_info(self, mock_imp: MagicMock) -> None:
        df = pd.DataFrame(
            {
                "info_type": ["catalog"],
                "collection_name": ["c"],
                "document_name": ["NA"],
                "info_value": ['{"k": 1}'],
            }
        )
        mock_table = MagicMock()
        mock_table.to_pandas.return_value = df
        mock_db = MagicMock()
        mock_db.table_names.return_value = [DEFAULT_DOCUMENT_INFO_COLLECTION]
        mock_db.open_table.return_value = mock_table
        mock_imp.return_value.connect.return_value = mock_db

        vdb = LanceDBVDB("t", "/x")
        assert vdb.get_document_info("catalog", "c", "NA") == {"k": 1}

    def test_get_catalog_metadata_delegates(self) -> None:
        vdb = LanceDBVDB("t", "/x")
        with patch.object(vdb, "get_document_info", return_value={"z": 1}) as mock_gdi:
            out = vdb.get_catalog_metadata("coll")
        mock_gdi.assert_called_once_with(
            info_type="catalog",
            collection_name="coll",
            document_name="NA",
        )
        assert out == {"z": 1}

    @patch(
        "nvidia_rag.utils.vdb.lancedb.lancedb_vdb.get_current_timestamp",
        return_value="ts",
    )
    def test_update_catalog_metadata(self, _ts: MagicMock) -> None:
        vdb = LanceDBVDB("t", "/x")
        with (
            patch.object(vdb, "get_catalog_metadata", return_value={"old": 1}),
            patch.object(vdb, "add_document_info") as mock_add,
        ):
            vdb.update_catalog_metadata("coll", {"new": 2})
        mock_add.assert_called_once()
        stored = mock_add.call_args[1]["info_value"]
        assert stored["new"] == 2
        assert stored["last_updated"] == "ts"

    def test_get_document_catalog_metadata(self) -> None:
        vdb = LanceDBVDB("t", "/x")
        with patch.object(
            vdb,
            "get_document_info",
            return_value={"description": "d", "tags": ["t"], "extra": 1},
        ):
            out = vdb.get_document_catalog_metadata("c", "doc.pdf")
        assert out == {"description": "d", "tags": ["t"]}

    def test_update_document_catalog_metadata_merges(self) -> None:
        vdb = LanceDBVDB("t", "/x")
        with (
            patch.object(
                vdb,
                "get_document_info",
                return_value={"description": "old", "tags": [], "keep": True},
            ),
            patch.object(vdb, "add_document_info") as mock_add,
        ):
            vdb.update_document_catalog_metadata(
                "c", "d.pdf", {"description": "new", "tags": ["x"]}
            )
        merged = mock_add.call_args[1]["info_value"]
        assert merged["description"] == "new"
        assert merged["tags"] == ["x"]
        assert merged["keep"] is True


class TestInternalHelpers(unittest.TestCase):
    """_delete_from_system_table, _get_document_info_map."""

    @patch("nvidia_rag.utils.vdb.lancedb.lancedb_vdb._import_lancedb")
    def test_delete_from_system_table(self, mock_imp: MagicMock) -> None:
        mock_table = MagicMock()
        mock_db = MagicMock()
        mock_db.table_names.return_value = ["metadata_schema"]
        mock_db.open_table.return_value = mock_table
        mock_imp.return_value.connect.return_value = mock_db

        vdb = LanceDBVDB("t", "/x")
        vdb._delete_from_system_table("metadata_schema", "collection_name", "c1")
        mock_table.delete.assert_called_once()

    @patch("nvidia_rag.utils.vdb.lancedb.lancedb_vdb._import_lancedb")
    def test_get_document_info_map(self, mock_imp: MagicMock) -> None:
        df = pd.DataFrame(
            {
                "info_type": ["document", "document"],
                "collection_name": ["c", "c"],
                "document_name": ["a.pdf", "b.pdf"],
                "info_value": ['{"x": 1}', "not-json"],
            }
        )
        mock_table = MagicMock()
        mock_table.to_pandas.return_value = df
        mock_db = MagicMock()
        mock_db.table_names.return_value = [DEFAULT_DOCUMENT_INFO_COLLECTION]
        mock_db.open_table.return_value = mock_table
        mock_imp.return_value.connect.return_value = mock_db

        vdb = LanceDBVDB("t", "/x")
        m = vdb._get_document_info_map("c")
        assert m["a.pdf"] == {"x": 1}
        assert m["b.pdf"] == {}


if __name__ == "__main__":
    unittest.main()

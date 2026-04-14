# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for NemoRetrieverHandler."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from nvidia_rag.ingestor_server.nemo_retriever.handler import NemoRetrieverHandler
from nvidia_rag.ingestor_server.nemo_retriever.ingest_schema_manager import (
    IngestSchemaManager,
)
from nvidia_rag.utils.configuration import NvidiaRAGConfig, NvIngestConfig


def _nv_ingest_for_handler(**kwargs: object) -> NvIngestConfig:
    base: dict[str, object] = {
        "page_elements_invoke_url": None,
        "graphic_elements_invoke_url": None,
        "ocr_invoke_url": None,
        "table_structure_invoke_url": None,
        "extract_images": False,
        "nrl_run_mode": "inprocess",
    }
    base.update(kwargs)
    return NvIngestConfig(**base)


@pytest.fixture
def nemo_config() -> NvidiaRAGConfig:
    return NvidiaRAGConfig(nv_ingest=_nv_ingest_for_handler())


@pytest.fixture
def handler(nemo_config: NvidiaRAGConfig) -> NemoRetrieverHandler:
    h = NemoRetrieverHandler(nemo_config)
    yield h
    h._executor.shutdown(wait=False)


class TestNemoRetrieverHandlerBuildIngestor:
    @patch("nvidia_rag.ingestor_server.nemo_retriever.handler.GraphIngestor")
    def test_full_pipeline_calls_stages(
        self, mock_gi_class: MagicMock, nemo_config: NvidiaRAGConfig
    ) -> None:
        mock_chain = MagicMock()
        mock_gi_class.return_value = mock_chain
        for name in ("files", "extract", "split", "caption", "store", "embed"):
            getattr(mock_chain, name).return_value = mock_chain

        nv = _nv_ingest_for_handler(extract_images=True)
        config = NvidiaRAGConfig(nv_ingest=nv)
        h = NemoRetrieverHandler(config)
        try:
            vdb = MagicMock()
            vdb.collection_name = "c1"
            h._build_ingestor(
                ["/tmp/a.pdf"],
                split_options=None,
                extract_override=None,
                vdb_op=vdb,
                store_images=True,
            )
        finally:
            h._executor.shutdown(wait=False)

        mock_gi_class.assert_called_once_with(run_mode="inprocess")
        mock_chain.files.assert_called_once_with(["/tmp/a.pdf"])
        mock_chain.extract.assert_called_once()
        mock_chain.split.assert_called_once()
        mock_chain.caption.assert_called_once()
        mock_chain.store.assert_called_once()
        mock_chain.embed.assert_called_once()

    @patch("nvidia_rag.ingestor_server.nemo_retriever.handler.GraphIngestor")
    def test_skips_caption_store_embed_when_configured(
        self, mock_gi_class: MagicMock, nemo_config: NvidiaRAGConfig
    ) -> None:
        mock_chain = MagicMock()
        mock_gi_class.return_value = mock_chain
        for name in ("files", "extract", "split"):
            getattr(mock_chain, name).return_value = mock_chain

        h = NemoRetrieverHandler(nemo_config)
        try:
            h._build_ingestor(
                ["/x.pdf"],
                split_options=None,
                extract_override=None,
                vdb_op=None,
                store_images=False,
            )
        finally:
            h._executor.shutdown(wait=False)

        mock_chain.caption.assert_not_called()
        mock_chain.store.assert_not_called()
        mock_chain.embed.assert_not_called()

    @patch("nvidia_rag.ingestor_server.nemo_retriever.handler.GraphIngestor")
    def test_shallow_ingestor_only_extract(
        self, mock_gi_class: MagicMock, nemo_config: NvidiaRAGConfig
    ) -> None:
        mock_chain = MagicMock()
        mock_gi_class.return_value = mock_chain
        mock_chain.files.return_value = mock_chain
        mock_chain.extract.return_value = mock_chain

        h = NemoRetrieverHandler(nemo_config)
        try:
            h._build_shallow_ingestor(["/y.pdf"])
        finally:
            h._executor.shutdown(wait=False)

        mock_chain.files.assert_called_once_with(["/y.pdf"])
        mock_chain.extract.assert_called_once()
        mock_chain.split.assert_not_called()


class TestNemoRetrieverHandlerRunSync:
    def test_inprocess_returns_dataframe(self, handler: NemoRetrieverHandler) -> None:
        df = pd.DataFrame({"path": ["/a"], "text": ["t"]})
        ingestor = MagicMock()
        ingestor.ingest.return_value = df
        out = handler._run_sync(ingestor)
        pd.testing.assert_frame_equal(out, df)

    def test_batch_materializes_ray_dataset(
        self, handler: NemoRetrieverHandler
    ) -> None:
        handler._run_mode = "batch"
        records = [{"path": "/b", "text": "x"}]
        dataset = MagicMock()
        dataset.take_all.return_value = records
        ingestor = MagicMock()
        ingestor.ingest.return_value = dataset

        out = handler._run_sync(ingestor)
        assert isinstance(out, pd.DataFrame)
        assert len(out) == 1
        assert out.iloc[0]["path"] == "/b"


class TestNemoRetrieverHandlerAsync:
    @pytest.mark.asyncio
    @patch.object(NemoRetrieverHandler, "_run_sync")
    @patch.object(NemoRetrieverHandler, "_build_ingestor")
    async def test_ingest_returns_schema_manager(
        self,
        mock_build: MagicMock,
        mock_run: MagicMock,
        nemo_config: NvidiaRAGConfig,
    ) -> None:
        df = pd.DataFrame({"c": [1]})
        mock_run.return_value = df
        mock_gi = MagicMock()
        mock_build.return_value = mock_gi

        h = NemoRetrieverHandler(nemo_config)
        try:
            vdb = MagicMock()
            result = await h.ingest(["/f.pdf"], vdb)
        finally:
            h._executor.shutdown(wait=False)

        assert isinstance(result, IngestSchemaManager)
        assert result.row_count() == 1
        mock_build.assert_called_once()
        mock_run.assert_called_once_with(mock_gi)

    @pytest.mark.asyncio
    @patch.object(NemoRetrieverHandler, "_run_sync")
    @patch.object(NemoRetrieverHandler, "_build_shallow_ingestor")
    async def test_ingest_shallow(
        self,
        mock_shallow: MagicMock,
        mock_run: MagicMock,
        nemo_config: NvidiaRAGConfig,
    ) -> None:
        mock_run.return_value = pd.DataFrame({"x": [1]})
        mock_shallow.return_value = MagicMock()

        h = NemoRetrieverHandler(nemo_config)
        try:
            mgr = await h.ingest_shallow(["/z.pdf"])
        finally:
            h._executor.shutdown(wait=False)

        mock_shallow.assert_called_once_with(["/z.pdf"])
        assert isinstance(mgr, IngestSchemaManager)


class TestNemoRetrieverHandlerInit:
    def test_run_mode_from_config(self) -> None:
        nv = _nv_ingest_for_handler(nrl_run_mode="batch")
        h = NemoRetrieverHandler(NvidiaRAGConfig(nv_ingest=nv))
        try:
            assert h._run_mode == "batch"
        finally:
            h._executor.shutdown(wait=False)

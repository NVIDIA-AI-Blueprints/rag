# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for nemo_retriever.params adapter functions."""

from unittest.mock import MagicMock

from pydantic import SecretStr

from nvidia_rag.ingestor_server.nemo_retriever.params import (
    make_caption_params,
    make_embed_params,
    make_extract_params,
    make_split_params,
    make_store_params,
)
from nvidia_rag.utils.configuration import (
    EmbeddingConfig,
    ObjectStoreConfig,
    NvidiaRAGConfig,
    NvIngestConfig,
)
from nvidia_rag.utils.object_store import DEFAULT_BUCKET_NAME


def _base_nv_ingest(**kwargs: object) -> NvIngestConfig:
    """NvIngestConfig with NIM URLs cleared so extract params stay deterministic."""
    defaults: dict[str, object] = {
        "page_elements_invoke_url": None,
        "graphic_elements_invoke_url": None,
        "ocr_invoke_url": None,
        "table_structure_invoke_url": None,
    }
    defaults.update(kwargs)
    return NvIngestConfig(**defaults)


class TestMakeExtractParams:
    def test_maps_config_and_batch_tuning(self) -> None:
        nv = _base_nv_ingest(
            extract_text=True,
            extract_tables=False,
            extract_charts=True,
            extract_infographics=False,
            extract_images=True,
            extract_page_as_image=True,
        )
        config = NvidiaRAGConfig(nv_ingest=nv)
        ep = make_extract_params(config)

        assert ep.extract_text is True
        assert ep.extract_tables is False
        assert ep.extract_charts is True
        assert ep.extract_infographics is False
        assert ep.extract_images is True
        assert ep.extract_page_as_image is True
        assert ep.table_output_format == "markdown"
        assert ep.batch_tuning.pdf_extract_workers == 9

    def test_conditional_invoke_urls(self) -> None:
        nv = NvIngestConfig(
            page_elements_invoke_url="http://page.example",
            graphic_elements_invoke_url="http://graphic.example",
            ocr_invoke_url="http://ocr.example",
            table_structure_invoke_url="http://table.example",
        )
        config = NvidiaRAGConfig(nv_ingest=nv)
        ep = make_extract_params(config)

        assert ep.page_elements_invoke_url == "http://page.example"
        assert ep.graphic_elements_invoke_url == "http://graphic.example"
        assert ep.ocr_invoke_url == "http://ocr.example"
        assert ep.table_structure_invoke_url == "http://table.example"

    def test_override_merges_last(self) -> None:
        nv = _base_nv_ingest(extract_images=True)
        config = NvidiaRAGConfig(nv_ingest=nv)
        ep = make_extract_params(config, override={"extract_images": False})
        assert ep.extract_images is False


class TestMakeSplitParams:
    def test_chunk_mapping(self) -> None:
        nv = _base_nv_ingest(chunk_size=512, chunk_overlap=64)
        config = NvidiaRAGConfig(nv_ingest=nv)
        sp = make_split_params(config)
        assert sp.max_tokens == 512
        assert sp.overlap_tokens == 64


class TestMakeEmbedParams:
    def test_model_server_and_modality(self) -> None:
        emb = EmbeddingConfig(model_name="m1", server_url="http://embed.example/v1")
        nv = _base_nv_ingest(structured_elements_modality="text_image")
        config = NvidiaRAGConfig(embeddings=emb, nv_ingest=nv)
        ep = make_embed_params(config)
        assert ep.model_name == "m1"
        assert ep.embed_invoke_url == "http://embed.example/v1"
        assert ep.embed_modality == "text_image"

    def test_empty_optional_fields(self) -> None:
        emb = EmbeddingConfig(model_name="", server_url="")
        nv = _base_nv_ingest(structured_elements_modality="")
        config = NvidiaRAGConfig(embeddings=emb, nv_ingest=nv)
        ep = make_embed_params(config)
        assert ep.model_name is None
        assert ep.embed_invoke_url is None


class TestMakeCaptionParams:
    def test_model_and_endpoint(self) -> None:
        nv = _base_nv_ingest(
            caption_model_name="cap-model",
            caption_endpoint_url="http://caption.example",
        )
        config = NvidiaRAGConfig(nv_ingest=nv)
        cp = make_caption_params(config)
        assert cp.model_name == "cap-model"
        assert cp.endpoint_url == "http://caption.example"


class TestMakeStoreParams:
    def test_storage_uri_and_options(self) -> None:
        object_store = ObjectStoreConfig(
            endpoint="minio:9000",
            access_key=SecretStr("ak"),
            secret_key=SecretStr("sk"),
        )
        config = NvidiaRAGConfig(object_store=object_store)
        vdb_op = MagicMock()
        vdb_op.collection_name = "my_collection"

        sp = make_store_params(config, vdb_op)

        expected_uri = f"s3://{DEFAULT_BUCKET_NAME}/my_collection/images"
        assert sp.storage_uri == expected_uri
        assert sp.public_base_url == expected_uri
        assert sp.storage_options["key"] == "ak"
        assert sp.storage_options["secret"] == "sk"
        assert (
            sp.storage_options["client_kwargs"]["endpoint_url"] == "http://minio:9000"
        )

    def test_filesystem_storage_uri(self, tmp_path) -> None:
        object_store = ObjectStoreConfig(
            backend="filesystem",
            local_path=str(tmp_path / "object-store"),
        )
        config = NvidiaRAGConfig(object_store=object_store)
        vdb_op = MagicMock()
        vdb_op.collection_name = "my_collection"

        sp = make_store_params(config, vdb_op)

        expected_uri = (
            tmp_path / "object-store" / DEFAULT_BUCKET_NAME / "my_collection" / "images"
        ).resolve().as_uri()
        assert sp.storage_uri == expected_uri
        assert sp.public_base_url == expected_uri
        assert sp.storage_options == {}

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

"""Unit tests for NvidiaRAGRetriever."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from langchain_core.documents import Document

from nvidia_rag.retriever import NvidiaRAGRetriever


# ── Fixtures ──────────────────────────────────────────────────────────────

SAMPLE_SEARCH_RESPONSE = {
    "total_results": 2,
    "results": [
        {
            "document_id": "doc-001",
            "content": "CUDA is a parallel computing platform.",
            "document_name": "cuda_guide.pdf",
            "document_type": "text",
            "score": 0.95,
            "metadata": {
                "language": "en",
                "date_created": "2024-01-15",
                "last_modified": "2024-06-01",
                "page_number": 3,
                "description": "Introduction to CUDA",
                "height": 0,
                "width": 0,
                "location": [],
                "location_max_dimensions": [],
                "content_metadata": {"author": "NVIDIA"},
            },
        },
        {
            "document_id": "doc-002",
            "content": "TensorRT optimizes deep learning inference.",
            "document_name": "tensorrt_docs.pdf",
            "document_type": "text",
            "score": 0.87,
            "metadata": {
                "language": "en",
                "date_created": "2024-03-10",
                "last_modified": "2024-05-20",
                "page_number": 1,
                "description": "",
                "height": 0,
                "width": 0,
                "location": [],
                "location_max_dimensions": [],
                "content_metadata": {},
            },
        },
    ],
}

EMPTY_SEARCH_RESPONSE = {"total_results": 0, "results": []}


@pytest.fixture
def retriever():
    return NvidiaRAGRetriever(base_url="http://localhost:8081")


@pytest.fixture
def retriever_with_options():
    return NvidiaRAGRetriever(
        base_url="http://localhost:8081/",
        api_key="test-api-key",
        collection_name="my_collection",
        top_k=5,
        vdb_top_k=50,
        filters='content_metadata["author"] == "NVIDIA"',
        enable_reranker=True,
        enable_query_rewriting=False,
        embedding_model="llama-3.2-nv-embedqa-1b-v2",
        reranker_model="llama-3.2-nv-rerankqa-1b-v2",
    )


# ── Constructor Tests ─────────────────────────────────────────────────────


class TestConstructor:
    def test_minimal_constructor(self, retriever):
        assert retriever.base_url == "http://localhost:8081"
        assert retriever.api_key is None
        assert retriever.collection_name is None
        assert retriever.top_k == 10
        assert retriever.vdb_top_k == 100

    def test_trailing_slash_stripped(self, retriever_with_options):
        assert retriever_with_options.base_url == "http://localhost:8081"

    def test_api_key_stored_as_secret(self, retriever_with_options):
        assert retriever_with_options.api_key is not None
        assert retriever_with_options.api_key.get_secret_value() == "test-api-key"

    def test_all_options_set(self, retriever_with_options):
        assert retriever_with_options.collection_name == "my_collection"
        assert retriever_with_options.top_k == 5
        assert retriever_with_options.vdb_top_k == 50
        assert retriever_with_options.enable_reranker is True
        assert retriever_with_options.enable_query_rewriting is False

    def test_base_url_required(self):
        with pytest.raises(Exception):
            NvidiaRAGRetriever()

    def test_top_k_validation(self):
        with pytest.raises(Exception):
            NvidiaRAGRetriever(base_url="http://localhost:8081", top_k=0)

        with pytest.raises(Exception):
            NvidiaRAGRetriever(base_url="http://localhost:8081", top_k=100)


# ── Payload Construction Tests ────────────────────────────────────────────


class TestPayloadConstruction:
    def test_minimal_payload(self, retriever):
        payload = retriever._build_payload("What is CUDA?")
        assert payload == {
            "query": "What is CUDA?",
            "reranker_top_k": 10,
            "vdb_top_k": 100,
        }

    def test_full_payload(self, retriever_with_options):
        payload = retriever_with_options._build_payload("Tell me about GPUs")
        assert payload["query"] == "Tell me about GPUs"
        assert payload["reranker_top_k"] == 5
        assert payload["vdb_top_k"] == 50
        assert payload["collection_names"] == ["my_collection"]
        assert payload["filter_expr"] == 'content_metadata["author"] == "NVIDIA"'
        assert payload["enable_reranker"] is True
        assert payload["enable_query_rewriting"] is False
        assert payload["embedding_model"] == "llama-3.2-nv-embedqa-1b-v2"
        assert payload["reranker_model"] == "llama-3.2-nv-rerankqa-1b-v2"

    def test_none_options_excluded(self, retriever):
        payload = retriever._build_payload("test")
        assert "collection_names" not in payload
        assert "filter_expr" not in payload
        assert "enable_reranker" not in payload
        assert "enable_query_rewriting" not in payload
        assert "embedding_model" not in payload
        assert "reranker_model" not in payload


# ── Headers Tests ─────────────────────────────────────────────────────────


class TestHeaders:
    def test_headers_without_auth(self, retriever):
        headers = retriever._build_headers()
        assert headers["Content-Type"] == "application/json"
        assert headers["Accept"] == "application/json"
        assert "Authorization" not in headers

    def test_headers_with_auth(self, retriever_with_options):
        headers = retriever_with_options._build_headers()
        assert headers["Authorization"] == "Bearer test-api-key"


# ── Response Parsing Tests ────────────────────────────────────────────────


class TestResponseParsing:
    def test_parse_standard_response(self):
        docs = NvidiaRAGRetriever._parse_response(SAMPLE_SEARCH_RESPONSE)
        assert len(docs) == 2

        doc0 = docs[0]
        assert isinstance(doc0, Document)
        assert doc0.page_content == "CUDA is a parallel computing platform."
        assert doc0.metadata["source"] == "cuda_guide.pdf"
        assert doc0.metadata["document_id"] == "doc-001"
        assert doc0.metadata["document_type"] == "text"
        assert doc0.metadata["score"] == 0.95
        assert doc0.metadata["page_number"] == 3
        assert doc0.metadata["language"] == "en"
        assert doc0.metadata["content_metadata"] == {"author": "NVIDIA"}

    def test_parse_empty_response(self):
        docs = NvidiaRAGRetriever._parse_response(EMPTY_SEARCH_RESPONSE)
        assert docs == []

    def test_parse_response_missing_fields(self):
        data = {
            "total_results": 1,
            "results": [
                {
                    "content": "Some content",
                }
            ],
        }
        docs = NvidiaRAGRetriever._parse_response(data)
        assert len(docs) == 1
        assert docs[0].page_content == "Some content"
        assert docs[0].metadata["source"] == ""
        assert docs[0].metadata["score"] == 0.0

    def test_parse_response_with_location(self):
        data = {
            "total_results": 1,
            "results": [
                {
                    "content": "A chart",
                    "document_type": "chart",
                    "metadata": {
                        "location": [10.0, 20.0, 100.0, 200.0],
                        "location_max_dimensions": [612, 792],
                    },
                }
            ],
        }
        docs = NvidiaRAGRetriever._parse_response(data)
        assert docs[0].metadata["location"] == [10.0, 20.0, 100.0, 200.0]
        assert docs[0].metadata["location_max_dimensions"] == [612, 792]


# ── Sync Retrieval Tests ─────────────────────────────────────────────────


class TestSyncRetrieval:
    def test_invoke_success(self, retriever):
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_SEARCH_RESPONSE
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            retriever._sync_client, "post", return_value=mock_response
        ) as mock_post:
            docs = retriever.invoke("What is CUDA?")

            mock_post.assert_called_once_with(
                "/v1/search",
                json={
                    "query": "What is CUDA?",
                    "reranker_top_k": 10,
                    "vdb_top_k": 100,
                },
            )
            assert len(docs) == 2
            assert docs[0].page_content == "CUDA is a parallel computing platform."

    def test_invoke_empty_results(self, retriever):
        mock_response = MagicMock()
        mock_response.json.return_value = EMPTY_SEARCH_RESPONSE
        mock_response.raise_for_status = MagicMock()

        with patch.object(retriever._sync_client, "post", return_value=mock_response):
            docs = retriever.invoke("nonexistent topic")
            assert docs == []

    def test_invoke_http_error(self, retriever):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error",
            request=MagicMock(),
            response=MagicMock(status_code=500),
        )

        with patch.object(retriever._sync_client, "post", return_value=mock_response):
            with pytest.raises(httpx.HTTPStatusError):
                retriever.invoke("test query")

    def test_invoke_connection_error(self, retriever):
        with patch.object(
            retriever._sync_client,
            "post",
            side_effect=httpx.ConnectError("Connection refused"),
        ):
            with pytest.raises(httpx.ConnectError):
                retriever.invoke("test query")


# ── Async Retrieval Tests ────────────────────────────────────────────────


class TestAsyncRetrieval:
    @pytest.mark.asyncio
    async def test_ainvoke_success(self, retriever):
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_SEARCH_RESPONSE
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            retriever._async_client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_post:
            docs = await retriever.ainvoke("What is CUDA?")

            mock_post.assert_called_once_with(
                "/v1/search",
                json={
                    "query": "What is CUDA?",
                    "reranker_top_k": 10,
                    "vdb_top_k": 100,
                },
            )
            assert len(docs) == 2

    @pytest.mark.asyncio
    async def test_ainvoke_http_error(self, retriever):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found",
            request=MagicMock(),
            response=MagicMock(status_code=404),
        )

        with patch.object(
            retriever._async_client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            with pytest.raises(httpx.HTTPStatusError):
                await retriever.ainvoke("test query")


# ── Lifecycle Tests ──────────────────────────────────────────────────────


class TestLifecycle:
    def test_close(self, retriever):
        retriever.close()
        assert retriever._sync_client.is_closed

    @pytest.mark.asyncio
    async def test_aclose(self, retriever):
        await retriever.aclose()
        assert retriever._async_client.is_closed


# ── Filter Tests ─────────────────────────────────────────────────────────


class TestFilters:
    def test_string_filter(self):
        r = NvidiaRAGRetriever(
            base_url="http://localhost:8081",
            filters='content_metadata["category"] == "science"',
        )
        payload = r._build_payload("test")
        assert payload["filter_expr"] == 'content_metadata["category"] == "science"'

    def test_list_filter(self):
        filter_list = [{"term": {"category": "science"}}]
        r = NvidiaRAGRetriever(
            base_url="http://localhost:8081",
            filters=filter_list,
        )
        payload = r._build_payload("test")
        assert payload["filter_expr"] == filter_list

    def test_empty_filter_excluded(self):
        r = NvidiaRAGRetriever(base_url="http://localhost:8081", filters="")
        payload = r._build_payload("test")
        assert "filter_expr" not in payload

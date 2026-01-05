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

"""Unit tests for summarization utilities."""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from langchain_core.documents import Document

from nvidia_rag.utils.configuration import NvidiaRAGConfig
from nvidia_rag.utils.summarization import (
    _batch_summaries_by_length,
    _combine_summaries_batch,
    _extract_content_from_element,
    _generate_single_document_summary,
    _get_summary_llm,
    _prepare_single_document,
    _process_single_file_summary,
    _store_summary_in_minio,
    _summarize_hierarchical,
    _summarize_iterative,
    _summarize_single_pass,
    _update_file_progress,
    generate_document_summaries,
    get_summarization_semaphore,
    matches_page_filter,
)


class TestMatchesPageFilterErrorHandling:
    """Test error handling in matches_page_filter"""

    def test_matches_page_filter_invalid_range_format(self):
        """Test matches_page_filter with invalid range format"""
        result = matches_page_filter(5, [[1, 2, 3]], total_pages=10)
        assert result is False

    def test_matches_page_filter_non_list_non_string(self):
        """Test matches_page_filter with invalid type"""
        result = matches_page_filter(5, 123, total_pages=10)
        assert result is False


class TestGetSummarizationSemaphoreErrorHandling:
    """Test error handling in get_summarization_semaphore"""

    def test_get_summarization_semaphore_no_event_loop(self):
        """Test get_summarization_semaphore raises RuntimeError when no event loop"""
        with pytest.raises(RuntimeError, match="No running event loop"):
            get_summarization_semaphore()


class TestExtractContentFromElement:
    """Test _extract_content_from_element function"""

    def test_extract_content_text_type(self):
        """Test extracting content from text element"""
        config = Mock()
        elem = {
            "document_type": "text",
            "metadata": {"content": "Test text content"},
        }
        result = _extract_content_from_element(elem, config)
        assert result == "Test text content"

    def test_extract_content_structured_table_enabled(self):
        """Test extracting content from structured table when enabled"""
        config = Mock()
        config.nv_ingest.extract_tables = True
        elem = {
            "document_type": "structured",
            "metadata": {
                "table_metadata": {"table_content": "Table content"},
                "content_metadata": {"subtype": "table"},
            },
        }
        result = _extract_content_from_element(elem, config)
        assert result == "Table content"

    def test_extract_content_structured_table_disabled(self):
        """Test extracting content from structured table when disabled"""
        config = Mock()
        config.nv_ingest.extract_tables = False
        elem = {
            "document_type": "structured",
            "metadata": {
                "table_metadata": {"table_content": "Table content"},
                "content_metadata": {"subtype": "table"},
            },
        }
        result = _extract_content_from_element(elem, config)
        assert result is None

    def test_extract_content_structured_chart_enabled(self):
        """Test extracting content from structured chart when enabled"""
        config = Mock()
        config.nv_ingest.extract_charts = True
        elem = {
            "document_type": "structured",
            "metadata": {
                "table_metadata": {"table_content": "Chart content"},
                "content_metadata": {"subtype": "chart"},
            },
        }
        result = _extract_content_from_element(elem, config)
        assert result == "Chart content"

    def test_extract_content_structured_chart_disabled(self):
        """Test extracting content from structured chart when disabled"""
        config = Mock()
        config.nv_ingest.extract_charts = False
        elem = {
            "document_type": "structured",
            "metadata": {
                "table_metadata": {"table_content": "Chart content"},
                "content_metadata": {"subtype": "chart"},
            },
        }
        result = _extract_content_from_element(elem, config)
        assert result is None

    def test_extract_content_image_enabled(self):
        """Test extracting content from image when enabled"""
        config = Mock()
        config.nv_ingest.extract_images = True
        elem = {
            "document_type": "image",
            "metadata": {"image_metadata": {"caption": "Image caption"}},
        }
        result = _extract_content_from_element(elem, config)
        assert result == "Image caption"

    def test_extract_content_image_disabled(self):
        """Test extracting content from image when disabled"""
        config = Mock()
        config.nv_ingest.extract_images = False
        elem = {
            "document_type": "image",
            "metadata": {"image_metadata": {"caption": "Image caption"}},
        }
        result = _extract_content_from_element(elem, config)
        assert result is None

    def test_extract_content_audio(self):
        """Test extracting content from audio"""
        config = Mock()
        elem = {
            "document_type": "audio",
            "metadata": {"audio_metadata": {"audio_transcript": "Audio transcript"}},
        }
        result = _extract_content_from_element(elem, config)
        assert result == "Audio transcript"

    def test_extract_content_unknown_type(self):
        """Test extracting content from unknown type"""
        config = Mock()
        elem = {"document_type": "unknown", "metadata": {}}
        result = _extract_content_from_element(elem, config)
        assert result is None


class TestPrepareSingleDocument:
    """Test _prepare_single_document function"""

    @pytest.mark.asyncio
    async def test_prepare_single_document_success(self):
        """Test successful document preparation"""
        config = Mock()
        result_element = {
            "metadata": {
                "source_metadata": {"source_id": "/path/to/file.pdf"},
            },
        }
        results = [
            [
                {
                    "metadata": {
                        "source_metadata": {"source_id": "/path/to/file.pdf"},
                        "content_metadata": {"page_number": 1},
                        "content": "Page 1 content",
                    },
                    "document_type": "text",
                },
                {
                    "metadata": {
                        "source_metadata": {"source_id": "/path/to/file.pdf"},
                        "content_metadata": {"page_number": 2},
                        "content": "Page 2 content",
                    },
                    "document_type": "text",
                },
            ],
        ]

        with patch(
            "nvidia_rag.utils.summarization._extract_content_from_element"
        ) as mock_extract:
            mock_extract.side_effect = ["Page 1 content", "Page 2 content"]

            doc = await _prepare_single_document(
                result_element=result_element,
                results=results,
                collection_name="test_collection",
                config=config,
            )

            assert doc.page_content == "Page 1 content Page 2 content"
            assert doc.metadata["filename"] == "file.pdf"
            assert doc.metadata["collection_name"] == "test_collection"

    @pytest.mark.asyncio
    async def test_prepare_single_document_with_page_filter(self):
        """Test document preparation with page filter"""
        config = Mock()
        result_element = {
            "metadata": {
                "source_metadata": {"source_id": "/path/to/file.pdf"},
            },
        }
        results = [
            [
                {
                    "metadata": {
                        "source_metadata": {"source_id": "/path/to/file.pdf"},
                        "content_metadata": {"page_number": 1},
                        "content": "Page 1 content",
                    },
                    "document_type": "text",
                },
                {
                    "metadata": {
                        "source_metadata": {"source_id": "/path/to/file.pdf"},
                        "content_metadata": {"page_number": 2},
                        "content": "Page 2 content",
                    },
                    "document_type": "text",
                },
            ],
        ]

        with patch(
            "nvidia_rag.utils.summarization._extract_content_from_element"
        ) as mock_extract:
            mock_extract.side_effect = ["Page 1 content", "Page 2 content"]

            doc = await _prepare_single_document(
                result_element=result_element,
                results=results,
                collection_name="test_collection",
                config=config,
                page_filter=[[1, 1]],
            )

            assert "Page 1 content" in doc.page_content
            assert "Page 2 content" not in doc.page_content

    @pytest.mark.asyncio
    async def test_prepare_single_document_no_content(self):
        """Test document preparation with no content"""
        config = Mock()
        result_element = {
            "metadata": {
                "source_metadata": {"source_id": "/path/to/file.pdf"},
            },
        }
        results = [[]]

        with pytest.raises(ValueError, match="No content found"):
            await _prepare_single_document(
                result_element=result_element,
                results=results,
                collection_name="test_collection",
                config=config,
            )

    @pytest.mark.asyncio
    async def test_prepare_single_document_page_filter_no_match(self):
        """Test document preparation with page filter that matches nothing"""
        config = Mock()
        config.nv_ingest.extract_tables = True
        config.nv_ingest.extract_charts = True
        config.nv_ingest.extract_images = True
        result_element = {
            "metadata": {
                "source_metadata": {"source_id": "/path/to/file.pdf"},
            },
        }
        results = [
            [
                {
                    "metadata": {
                        "source_metadata": {"source_id": "/path/to/file.pdf"},
                        "content_metadata": {"page_number": 1},
                        "content": "Page 1 content",
                    },
                    "document_type": "text",
                },
                {
                    "metadata": {
                        "source_metadata": {"source_id": "/path/to/file.pdf"},
                        "content_metadata": {"page_number": 3},
                        "content": "Page 3 content",
                    },
                    "document_type": "text",
                },
            ],
        ]

        with patch(
            "nvidia_rag.utils.summarization._extract_content_from_element"
        ) as mock_extract:
            mock_extract.side_effect = ["Page 1 content", "Page 3 content"]

            with pytest.raises(ValueError, match=r"No content found.*page filter"):
                await _prepare_single_document(
                    result_element=result_element,
                    results=results,
                    collection_name="test_collection",
                    config=config,
                    page_filter="even",
                )


class TestGenerateSingleDocumentSummary:
    """Test _generate_single_document_summary function"""

    @pytest.mark.asyncio
    async def test_generate_single_document_summary_single_strategy(self):
        """Test single strategy summarization"""
        config = Mock()
        document = Document(
            page_content="Test content",
            metadata={"filename": "test.pdf"},
        )

        with patch(
            "nvidia_rag.utils.summarization._summarize_single_pass"
        ) as mock_single:
            mock_single.return_value = document
            result = await _generate_single_document_summary(
                document=document,
                config=config,
                summarization_strategy="single",
            )
            mock_single.assert_called_once()
            assert result == document

    @pytest.mark.asyncio
    async def test_generate_single_document_summary_iterative_strategy(self):
        """Test iterative strategy summarization"""
        config = Mock()
        document = Document(
            page_content="Test content",
            metadata={"filename": "test.pdf"},
        )

        with patch(
            "nvidia_rag.utils.summarization._summarize_iterative"
        ) as mock_iterative:
            mock_iterative.return_value = document
            result = await _generate_single_document_summary(
                document=document,
                config=config,
                summarization_strategy="iterative",
            )
            mock_iterative.assert_called_once()
            assert result == document

    @pytest.mark.asyncio
    async def test_generate_single_document_summary_hierarchical_strategy(self):
        """Test hierarchical strategy summarization"""
        config = Mock()
        document = Document(
            page_content="Test content",
            metadata={"filename": "test.pdf"},
        )

        with patch(
            "nvidia_rag.utils.summarization._summarize_hierarchical"
        ) as mock_hierarchical:
            mock_hierarchical.return_value = document
            result = await _generate_single_document_summary(
                document=document,
                config=config,
                summarization_strategy="hierarchical",
            )
            mock_hierarchical.assert_called_once()
            assert result == document

    @pytest.mark.asyncio
    async def test_generate_single_document_summary_default_strategy(self):
        """Test default strategy (iterative) summarization"""
        config = Mock()
        document = Document(
            page_content="Test content",
            metadata={"filename": "test.pdf"},
        )

        with patch(
            "nvidia_rag.utils.summarization._summarize_iterative"
        ) as mock_iterative:
            mock_iterative.return_value = document
            result = await _generate_single_document_summary(
                document=document,
                config=config,
                summarization_strategy=None,
            )
            mock_iterative.assert_called_once()
            assert result == document

    @pytest.mark.asyncio
    async def test_generate_single_document_summary_invalid_strategy(self):
        """Test invalid strategy raises ValueError"""
        config = Mock()
        document = Document(
            page_content="Test content",
            metadata={"filename": "test.pdf"},
        )

        with pytest.raises(ValueError, match="Unknown summarization_strategy"):
            await _generate_single_document_summary(
                document=document,
                config=config,
                summarization_strategy="invalid",
            )


class TestSummarizeSinglePass:
    """Test _summarize_single_pass function"""

    @pytest.mark.asyncio
    async def test_summarize_single_pass_success(self):
        """Test successful single pass summarization"""
        config = Mock()
        config.summarizer.max_chunk_length = 1000
        document = Document(
            page_content="Test content",
            metadata={"filename": "test.pdf"},
        )

        mock_llm = Mock()
        mock_chain = AsyncMock()
        mock_chain.ainvoke = AsyncMock(return_value="Summary text")

        with (
            patch("nvidia_rag.utils.summarization._get_summary_llm") as mock_get_llm,
            patch("nvidia_rag.utils.summarization.get_prompts") as mock_get_prompts,
            patch(
                "nvidia_rag.utils.summarization._create_llm_chains"
            ) as mock_create_chains,
            patch("nvidia_rag.utils.summarization._token_length") as mock_token_length,
        ):
            mock_get_llm.return_value = mock_llm
            mock_get_prompts.return_value = {}
            mock_create_chains.return_value = (mock_chain, None)
            mock_token_length.return_value = 100

            result = await _summarize_single_pass(
                document=document,
                config=config,
                is_shallow=False,
            )

            assert result.metadata["summary"] == "Summary text"
            mock_chain.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_summarize_single_pass_truncation(self):
        """Test single pass summarization with truncation"""
        config = Mock()
        config.summarizer.max_chunk_length = 100
        document = Document(
            page_content="a" * 10000,
            metadata={"filename": "test.pdf"},
        )

        mock_llm = Mock()
        mock_chain = AsyncMock()
        mock_chain.ainvoke = AsyncMock(return_value="Summary text")

        with (
            patch("nvidia_rag.utils.summarization._get_summary_llm") as mock_get_llm,
            patch("nvidia_rag.utils.summarization.get_prompts") as mock_get_prompts,
            patch(
                "nvidia_rag.utils.summarization._create_llm_chains"
            ) as mock_create_chains,
            patch("nvidia_rag.utils.summarization._token_length") as mock_token_length,
        ):
            mock_get_llm.return_value = mock_llm
            mock_get_prompts.return_value = {}
            mock_create_chains.return_value = (mock_chain, None)
            mock_token_length.return_value = 500

            result = await _summarize_single_pass(
                document=document,
                config=config,
                is_shallow=False,
            )

            assert result.metadata["summary"] == "Summary text"
            mock_chain.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_summarize_single_pass_with_progress_callback(self):
        """Test single pass summarization with progress callback"""
        config = Mock()
        config.summarizer.max_chunk_length = 1000
        document = Document(
            page_content="Test content",
            metadata={"filename": "test.pdf"},
        )

        mock_llm = Mock()
        mock_chain = AsyncMock()
        mock_chain.ainvoke = AsyncMock(return_value="Summary text")
        progress_callback = AsyncMock()

        with (
            patch("nvidia_rag.utils.summarization._get_summary_llm") as mock_get_llm,
            patch("nvidia_rag.utils.summarization.get_prompts") as mock_get_prompts,
            patch(
                "nvidia_rag.utils.summarization._create_llm_chains"
            ) as mock_create_chains,
            patch("nvidia_rag.utils.summarization._token_length") as mock_token_length,
        ):
            mock_get_llm.return_value = mock_llm
            mock_get_prompts.return_value = {}
            mock_create_chains.return_value = (mock_chain, None)
            mock_token_length.return_value = 100

            result = await _summarize_single_pass(
                document=document,
                config=config,
                progress_callback=progress_callback,
                is_shallow=False,
            )

            assert result.metadata["summary"] == "Summary text"
            assert progress_callback.call_count == 2


class TestSummarizeIterative:
    """Test _summarize_iterative function"""

    @pytest.mark.asyncio
    async def test_summarize_iterative_single_chunk(self):
        """Test iterative summarization with single chunk"""
        config = Mock()
        config.summarizer.max_chunk_length = 1000
        config.summarizer.chunk_overlap = 100
        document = Document(
            page_content="Test content",
            metadata={"filename": "test.pdf"},
        )

        mock_llm = Mock()
        mock_chain = AsyncMock()
        mock_chain.ainvoke = AsyncMock(return_value="Summary text")

        with (
            patch("nvidia_rag.utils.summarization._get_summary_llm") as mock_get_llm,
            patch("nvidia_rag.utils.summarization.get_prompts") as mock_get_prompts,
            patch(
                "nvidia_rag.utils.summarization._create_llm_chains"
            ) as mock_create_chains,
            patch("nvidia_rag.utils.summarization._token_length") as mock_token_length,
        ):
            mock_get_llm.return_value = mock_llm
            mock_get_prompts.return_value = {}
            mock_create_chains.return_value = (mock_chain, None)
            mock_token_length.return_value = 500

            progress_callback = AsyncMock()

            result = await _summarize_iterative(
                document=document,
                config=config,
                progress_callback=progress_callback,
                is_shallow=False,
            )

            assert result.metadata["summary"] == "Summary text"
            mock_chain.ainvoke.assert_called_once()
            assert progress_callback.call_count == 2

    @pytest.mark.asyncio
    async def test_summarize_iterative_multiple_chunks(self):
        """Test iterative summarization with multiple chunks"""
        config = Mock()
        config.summarizer.max_chunk_length = 100
        config.summarizer.chunk_overlap = 10
        document = Document(
            page_content="Chunk 1 content. Chunk 2 content. Chunk 3 content.",
            metadata={"filename": "test.pdf"},
        )

        mock_llm = Mock()
        mock_initial_chain = AsyncMock()
        mock_iterative_chain = AsyncMock()
        mock_initial_chain.ainvoke = AsyncMock(return_value="Summary 1")
        mock_iterative_chain.ainvoke = AsyncMock(
            side_effect=["Summary 1+2", "Final Summary"]
        )

        with (
            patch("nvidia_rag.utils.summarization._get_summary_llm") as mock_get_llm,
            patch("nvidia_rag.utils.summarization.get_prompts") as mock_get_prompts,
            patch(
                "nvidia_rag.utils.summarization._create_llm_chains"
            ) as mock_create_chains,
            patch("nvidia_rag.utils.summarization._token_length") as mock_token_length,
            patch(
                "nvidia_rag.utils.summarization._get_tokenizer"
            ) as mock_get_tokenizer,
            patch(
                "nvidia_rag.utils.summarization._split_text_into_chunks"
            ) as mock_split_chunks,
        ):
            mock_get_llm.return_value = mock_llm
            mock_get_prompts.return_value = {}
            mock_create_chains.return_value = (
                mock_initial_chain,
                mock_iterative_chain,
            )
            mock_token_length.return_value = 300
            mock_get_tokenizer.return_value = Mock()
            mock_split_chunks.return_value = [
                "Chunk 1 content.",
                "Chunk 2 content.",
                "Chunk 3 content.",
            ]

            progress_callback = AsyncMock()

            result = await _summarize_iterative(
                document=document,
                config=config,
                progress_callback=progress_callback,
                is_shallow=False,
            )

            assert result.metadata["summary"] == "Final Summary"
            assert mock_initial_chain.ainvoke.call_count == 1
            assert mock_iterative_chain.ainvoke.call_count == 2
            assert progress_callback.call_count == 4


class TestSummarizeHierarchical:
    """Test _summarize_hierarchical function"""

    @pytest.mark.asyncio
    async def test_summarize_hierarchical_single_chunk(self):
        """Test hierarchical summarization falls back to single pass for small docs"""
        config = Mock()
        config.summarizer.max_chunk_length = 1000
        config.summarizer.chunk_overlap = 100
        document = Document(
            page_content="Test content",
            metadata={"filename": "test.pdf"},
        )

        with (
            patch(
                "nvidia_rag.utils.summarization._summarize_single_pass"
            ) as mock_single,
            patch("nvidia_rag.utils.summarization._token_length") as mock_token_length,
        ):
            mock_token_length.return_value = 500
            mock_single.return_value = document

            result = await _summarize_hierarchical(
                document=document,
                config=config,
                is_shallow=False,
            )

            mock_single.assert_called_once()
            assert result == document

    @pytest.mark.asyncio
    async def test_summarize_hierarchical_multiple_chunks(self):
        """Test hierarchical summarization with multiple chunks"""
        config = Mock()
        config.summarizer.max_chunk_length = 100
        config.summarizer.chunk_overlap = 10
        document = Document(
            page_content="Chunk 1. Chunk 2. Chunk 3.",
            metadata={"filename": "test.pdf"},
        )

        mock_llm = Mock()
        mock_initial_chain = AsyncMock()
        mock_iterative_chain = AsyncMock()
        mock_initial_chain.ainvoke = AsyncMock(
            side_effect=["Summary 1", "Summary 2", "Summary 3"]
        )
        mock_iterative_chain.ainvoke = AsyncMock(return_value="Combined Summary")

        with (
            patch("nvidia_rag.utils.summarization._get_summary_llm") as mock_get_llm,
            patch("nvidia_rag.utils.summarization.get_prompts") as mock_get_prompts,
            patch(
                "nvidia_rag.utils.summarization._create_llm_chains"
            ) as mock_create_chains,
            patch("nvidia_rag.utils.summarization._token_length") as mock_token_length,
            patch(
                "nvidia_rag.utils.summarization._get_tokenizer"
            ) as mock_get_tokenizer,
            patch(
                "nvidia_rag.utils.summarization._split_text_into_chunks"
            ) as mock_split_chunks,
            patch(
                "nvidia_rag.utils.summarization._batch_summaries_by_length"
            ) as mock_batch,
            patch(
                "nvidia_rag.utils.summarization._combine_summaries_batch"
            ) as mock_combine,
        ):
            mock_get_llm.return_value = mock_llm
            mock_get_prompts.return_value = {}
            mock_create_chains.return_value = (
                mock_initial_chain,
                mock_iterative_chain,
            )
            mock_token_length.return_value = 300
            mock_get_tokenizer.return_value = Mock()
            mock_split_chunks.return_value = [
                "Chunk 1.",
                "Chunk 2.",
                "Chunk 3.",
            ]
            mock_batch.return_value = [["Summary 1", "Summary 2", "Summary 3"]]
            mock_combine.return_value = "Combined Summary"

            progress_callback = AsyncMock()

            result = await _summarize_hierarchical(
                document=document,
                config=config,
                progress_callback=progress_callback,
                is_shallow=False,
            )

            assert result.metadata["summary"] == "Combined Summary"
            assert progress_callback.call_count == 1


class TestBatchSummariesByLength:
    """Test _batch_summaries_by_length function"""

    def test_batch_summaries_by_length_single_batch(self):
        """Test batching summaries that fit in one batch"""
        summaries = ["Summary 1", "Summary 2", "Summary 3"]
        batches = _batch_summaries_by_length(summaries, max_chunk_chars=1000)
        assert len(batches) == 1
        assert batches[0] == summaries

    def test_batch_summaries_by_length_multiple_batches(self):
        """Test batching summaries that require multiple batches"""
        summaries = ["a" * 500, "b" * 500, "c" * 500]
        batches = _batch_summaries_by_length(summaries, max_chunk_chars=600)
        assert len(batches) >= 2

    def test_batch_summaries_by_length_empty_list(self):
        """Test batching empty list"""
        batches = _batch_summaries_by_length([], max_chunk_chars=1000)
        assert batches == []

    def test_batch_summaries_by_length_single_summary(self):
        """Test batching single summary"""
        batches = _batch_summaries_by_length(["Summary"], max_chunk_chars=1000)
        assert len(batches) == 1
        assert batches[0] == ["Summary"]


class TestCombineSummariesBatch:
    """Test _combine_summaries_batch function"""

    @pytest.mark.asyncio
    async def test_combine_summaries_batch_single_summary(self):
        """Test combining single summary returns it unchanged"""
        mock_chain = AsyncMock()
        result = await _combine_summaries_batch(
            summaries=["Single Summary"],
            iterative_chain=mock_chain,
            file_name="test.pdf",
            level=1,
            batch_idx=0,
        )
        assert result == "Single Summary"
        mock_chain.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_combine_summaries_batch_multiple_summaries(self):
        """Test combining multiple summaries"""
        mock_chain = AsyncMock()
        mock_chain.ainvoke = AsyncMock(side_effect=["Combined 1+2", "Final Combined"])

        result = await _combine_summaries_batch(
            summaries=["Summary 1", "Summary 2", "Summary 3"],
            iterative_chain=mock_chain,
            file_name="test.pdf",
            level=1,
            batch_idx=0,
        )

        assert result == "Final Combined"
        assert mock_chain.ainvoke.call_count == 2


class TestGetSummaryLLM:
    """Test _get_summary_llm function"""

    def test_get_summary_llm_with_endpoint(self):
        """Test getting LLM with server URL"""
        config = Mock()
        config.summarizer.model_name = "test_model"
        config.summarizer.temperature = 0.7
        config.summarizer.top_p = 0.9
        config.summarizer.get_api_key.return_value = "test_key"
        config.summarizer.server_url = "http://test.com"

        with patch("nvidia_rag.utils.summarization.get_llm") as mock_get_llm:
            mock_get_llm.return_value = Mock()
            _get_summary_llm(config)

            mock_get_llm.assert_called_once()
            call_kwargs = mock_get_llm.call_args[1]
            assert call_kwargs["llm_endpoint"] == "http://test.com"
            assert call_kwargs["model"] == "test_model"

    def test_get_summary_llm_without_endpoint(self):
        """Test getting LLM without server URL"""
        config = Mock()
        config.summarizer.model_name = "test_model"
        config.summarizer.temperature = 0.7
        config.summarizer.top_p = 0.9
        config.summarizer.get_api_key.return_value = "test_key"
        config.summarizer.server_url = None

        with patch("nvidia_rag.utils.summarization.get_llm") as mock_get_llm:
            mock_get_llm.return_value = Mock()
            _get_summary_llm(config)

            mock_get_llm.assert_called_once()
            call_kwargs = mock_get_llm.call_args[1]
            assert "llm_endpoint" not in call_kwargs


class TestUpdateFileProgress:
    """Test _update_file_progress function"""

    @pytest.mark.asyncio
    async def test_update_file_progress(self):
        """Test updating file progress"""
        with patch(
            "nvidia_rag.utils.summarization.SUMMARY_STATUS_HANDLER"
        ) as mock_handler:
            await _update_file_progress(
                collection_name="test_collection",
                file_name="test.pdf",
                current=5,
                total=10,
            )

            mock_handler.update_progress.assert_called_once()
            call_kwargs = mock_handler.update_progress.call_args[1]
            assert call_kwargs["collection_name"] == "test_collection"
            assert call_kwargs["file_name"] == "test.pdf"
            assert call_kwargs["status"] == "IN_PROGRESS"
            assert call_kwargs["progress"]["current"] == 5
            assert call_kwargs["progress"]["total"] == 10


class TestStoreSummaryInMinio:
    """Test _store_summary_in_minio function"""

    @pytest.mark.asyncio
    async def test_store_summary_in_minio_success(self):
        """Test successful summary storage in MinIO"""
        document = Document(
            page_content="Content",
            metadata={
                "summary": "Test summary",
                "filename": "test.pdf",
                "collection_name": "test_collection",
            },
        )

        mock_minio = Mock()
        mock_minio.put_payload = Mock()

        with (
            patch(
                "nvidia_rag.utils.summarization.get_minio_operator_instance"
            ) as mock_get_minio,
            patch(
                "nvidia_rag.utils.summarization.get_unique_thumbnail_id"
            ) as mock_get_id,
        ):
            mock_get_minio.return_value = mock_minio
            mock_get_id.return_value = "test_id"

            await _store_summary_in_minio(document)

            mock_get_id.assert_called_once()
            mock_minio.put_payload.assert_called_once()
            call_kwargs = mock_minio.put_payload.call_args[1]
            assert call_kwargs["payload"]["summary"] == "Test summary"
            assert call_kwargs["payload"]["file_name"] == "test.pdf"


class TestProcessSingleFileSummary:
    """Test _process_single_file_summary function"""

    @pytest.mark.asyncio
    async def test_process_single_file_summary_success(self):
        """Test successful single file summary processing"""
        config = Mock()
        config.summarizer.max_parallelization = 5
        file_data = {
            "file_name": "test.pdf",
            "result_element": {
                "metadata": {
                    "source_metadata": {"source_id": "/path/to/test.pdf"},
                },
            },
        }
        results = [[]]
        semaphore = asyncio.Semaphore(10)

        mock_document = Document(
            page_content="Content",
            metadata={"filename": "test.pdf", "summary": "Summary"},
        )

        with (
            patch(
                "nvidia_rag.utils.summarization.acquire_global_summary_slot"
            ) as mock_acquire,
            patch(
                "nvidia_rag.utils.summarization._prepare_single_document"
            ) as mock_prepare,
            patch(
                "nvidia_rag.utils.summarization._generate_single_document_summary"
            ) as mock_generate,
            patch("nvidia_rag.utils.summarization._store_summary_in_minio"),
            patch("nvidia_rag.utils.summarization.SUMMARY_STATUS_HANDLER"),
            patch("nvidia_rag.utils.summarization.release_global_summary_slot"),
        ):
            mock_acquire.return_value = True
            mock_prepare.return_value = mock_document
            mock_generate.return_value = mock_document

            result = await _process_single_file_summary(
                file_data=file_data,
                collection_name="test_collection",
                results=results,
                semaphore=semaphore,
                config=config,
            )

            assert result["status"] == "SUCCESS"
            assert result["file_name"] == "test.pdf"
            mock_prepare.assert_called_once()
            mock_generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_single_file_summary_failure(self):
        """Test single file summary processing with failure"""
        config = Mock()
        config.summarizer.max_parallelization = 5
        file_data = {
            "file_name": "test.pdf",
            "result_element": {
                "metadata": {
                    "source_metadata": {"source_id": "/path/to/test.pdf"},
                },
            },
        }
        results = [[]]
        semaphore = asyncio.Semaphore(10)

        with (
            patch(
                "nvidia_rag.utils.summarization.acquire_global_summary_slot"
            ) as mock_acquire,
            patch(
                "nvidia_rag.utils.summarization._prepare_single_document"
            ) as mock_prepare,
            patch("nvidia_rag.utils.summarization.SUMMARY_STATUS_HANDLER"),
            patch(
                "nvidia_rag.utils.summarization.release_global_summary_slot"
            ) as mock_release,
        ):
            mock_acquire.return_value = True
            mock_prepare.side_effect = ValueError("Test error")

            result = await _process_single_file_summary(
                file_data=file_data,
                collection_name="test_collection",
                results=results,
                semaphore=semaphore,
                config=config,
            )

            assert result["status"] == "FAILED"
            assert "error" in result
            mock_release.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_single_file_summary_slot_retry(self):
        """Test single file summary processing with slot retry"""
        config = Mock()
        config.summarizer.max_parallelization = 5
        file_data = {
            "file_name": "test.pdf",
            "result_element": {
                "metadata": {
                    "source_metadata": {"source_id": "/path/to/test.pdf"},
                },
            },
        }
        results = [[]]
        semaphore = asyncio.Semaphore(10)

        mock_document = Document(
            page_content="Content",
            metadata={"filename": "test.pdf", "summary": "Summary"},
        )

        with (
            patch(
                "nvidia_rag.utils.summarization.acquire_global_summary_slot"
            ) as mock_acquire,
            patch(
                "nvidia_rag.utils.summarization._prepare_single_document"
            ) as mock_prepare,
            patch(
                "nvidia_rag.utils.summarization._generate_single_document_summary"
            ) as mock_generate,
            patch("nvidia_rag.utils.summarization._store_summary_in_minio"),
            patch("nvidia_rag.utils.summarization.SUMMARY_STATUS_HANDLER"),
            patch("nvidia_rag.utils.summarization.release_global_summary_slot"),
            patch("asyncio.sleep") as mock_sleep,
        ):
            mock_acquire.side_effect = [False, False, True]
            mock_prepare.return_value = mock_document
            mock_generate.return_value = mock_document

            result = await _process_single_file_summary(
                file_data=file_data,
                collection_name="test_collection",
                results=results,
                semaphore=semaphore,
                config=config,
            )

            assert result["status"] == "SUCCESS"
            assert mock_acquire.call_count == 3
            assert mock_sleep.call_count == 2


class TestGenerateDocumentSummaries:
    """Test generate_document_summaries function"""

    @pytest.mark.asyncio
    async def test_generate_document_summaries_success(self):
        """Test successful document summaries generation"""
        results = [
            [
                {
                    "metadata": {
                        "source_metadata": {"source_id": "/path/to/file1.pdf"},
                    },
                },
            ],
            [
                {
                    "metadata": {
                        "source_metadata": {"source_id": "/path/to/file2.pdf"},
                    },
                },
            ],
        ]

        with (
            patch(
                "nvidia_rag.utils.summarization.get_summarization_semaphore"
            ) as mock_semaphore,
            patch(
                "nvidia_rag.utils.summarization._process_single_file_summary"
            ) as mock_process,
            patch(
                "nvidia_rag.utils.summarization.SUMMARY_STATUS_HANDLER"
            ) as mock_handler,
        ):
            mock_semaphore.return_value = asyncio.Semaphore(10)
            mock_handler.is_available.return_value = True
            mock_process.side_effect = [
                {"file_name": "file1.pdf", "status": "SUCCESS"},
                {"file_name": "file2.pdf", "status": "SUCCESS"},
            ]

            stats = await generate_document_summaries(
                results=results,
                collection_name="test_collection",
            )

            assert stats["total_files"] == 2
            assert stats["successful"] == 2
            assert stats["failed"] == 0

    @pytest.mark.asyncio
    async def test_generate_document_summaries_redis_unavailable(self):
        """Test document summaries generation when Redis is unavailable"""
        results = [
            [
                {
                    "metadata": {
                        "source_metadata": {"source_id": "/path/to/file1.pdf"},
                    },
                },
            ],
        ]

        with (
            patch(
                "nvidia_rag.utils.summarization.get_summarization_semaphore"
            ) as mock_semaphore,
            patch(
                "nvidia_rag.utils.summarization._process_single_file_summary"
            ) as mock_process,
            patch(
                "nvidia_rag.utils.summarization.SUMMARY_STATUS_HANDLER"
            ) as mock_handler,
        ):
            mock_semaphore.return_value = asyncio.Semaphore(10)
            mock_handler.is_available.return_value = False
            mock_process.return_value = {
                "file_name": "file1.pdf",
                "status": "SUCCESS",
            }

            stats = await generate_document_summaries(
                results=results,
                collection_name="test_collection",
            )

            assert stats["total_files"] == 1
            assert stats["successful"] == 1

    @pytest.mark.asyncio
    async def test_generate_document_summaries_no_files(self):
        """Test document summaries generation with no files"""
        results = []

        stats = await generate_document_summaries(
            results=results,
            collection_name="test_collection",
        )

        assert stats["total_files"] == 0
        assert stats["successful"] == 0
        assert stats["failed"] == 0

    @pytest.mark.asyncio
    async def test_generate_document_summaries_empty_results(self):
        """Test document summaries generation with empty result lists"""
        results = [[], []]

        stats = await generate_document_summaries(
            results=results,
            collection_name="test_collection",
        )

        assert stats["total_files"] == 0

    @pytest.mark.asyncio
    async def test_generate_document_summaries_with_failures(self):
        """Test document summaries generation with some failures"""
        results = [
            [
                {
                    "metadata": {
                        "source_metadata": {"source_id": "/path/to/file1.pdf"},
                    },
                },
            ],
            [
                {
                    "metadata": {
                        "source_metadata": {"source_id": "/path/to/file2.pdf"},
                    },
                },
            ],
        ]

        with (
            patch(
                "nvidia_rag.utils.summarization.get_summarization_semaphore"
            ) as mock_semaphore,
            patch(
                "nvidia_rag.utils.summarization._process_single_file_summary"
            ) as mock_process,
            patch("nvidia_rag.utils.summarization.SUMMARY_STATUS_HANDLER"),
        ):
            mock_semaphore.return_value = asyncio.Semaphore(10)
            mock_process.side_effect = [
                {"file_name": "file1.pdf", "status": "SUCCESS"},
                {"file_name": "file2.pdf", "status": "FAILED", "error": "Test error"},
            ]

            stats = await generate_document_summaries(
                results=results,
                collection_name="test_collection",
            )

            assert stats["total_files"] == 2
            assert stats["successful"] == 1
            assert stats["failed"] == 1

    @pytest.mark.asyncio
    async def test_generate_document_summaries_with_exceptions(self):
        """Test document summaries generation with exceptions"""
        results = [
            [
                {
                    "metadata": {
                        "source_metadata": {"source_id": "/path/to/file1.pdf"},
                    },
                },
            ],
        ]

        with (
            patch(
                "nvidia_rag.utils.summarization.get_summarization_semaphore"
            ) as mock_semaphore,
            patch(
                "nvidia_rag.utils.summarization._process_single_file_summary"
            ) as mock_process,
            patch("nvidia_rag.utils.summarization.SUMMARY_STATUS_HANDLER"),
        ):
            mock_semaphore.return_value = asyncio.Semaphore(10)
            mock_process.side_effect = ValueError("Unexpected error")

            stats = await generate_document_summaries(
                results=results,
                collection_name="test_collection",
            )

            assert stats["total_files"] == 1
            assert stats["successful"] == 0
            assert stats["failed"] == 1

    @pytest.mark.asyncio
    async def test_generate_document_summaries_with_page_filter(self):
        """Test document summaries generation with page filter"""
        results = [
            [
                {
                    "metadata": {
                        "source_metadata": {"source_id": "/path/to/file1.pdf"},
                    },
                },
            ],
        ]

        with (
            patch(
                "nvidia_rag.utils.summarization.get_summarization_semaphore"
            ) as mock_semaphore,
            patch(
                "nvidia_rag.utils.summarization._process_single_file_summary"
            ) as mock_process,
            patch("nvidia_rag.utils.summarization.SUMMARY_STATUS_HANDLER"),
        ):
            mock_semaphore.return_value = asyncio.Semaphore(10)
            mock_process.return_value = {
                "file_name": "file1.pdf",
                "status": "SUCCESS",
            }

            stats = await generate_document_summaries(
                results=results,
                collection_name="test_collection",
                page_filter=[[1, 5]],
            )

            assert stats["total_files"] == 1
            mock_process.assert_called_once()
            call_kwargs = mock_process.call_args[1]
            assert call_kwargs["page_filter"] == [[1, 5]]

    @pytest.mark.asyncio
    async def test_generate_document_summaries_with_strategy(self):
        """Test document summaries generation with strategy"""
        results = [
            [
                {
                    "metadata": {
                        "source_metadata": {"source_id": "/path/to/file1.pdf"},
                    },
                },
            ],
        ]

        with (
            patch(
                "nvidia_rag.utils.summarization.get_summarization_semaphore"
            ) as mock_semaphore,
            patch(
                "nvidia_rag.utils.summarization._process_single_file_summary"
            ) as mock_process,
            patch("nvidia_rag.utils.summarization.SUMMARY_STATUS_HANDLER"),
        ):
            mock_semaphore.return_value = asyncio.Semaphore(10)
            mock_process.return_value = {
                "file_name": "file1.pdf",
                "status": "SUCCESS",
            }

            stats = await generate_document_summaries(
                results=results,
                collection_name="test_collection",
                summarization_strategy="hierarchical",
            )

            assert stats["total_files"] == 1
            mock_process.assert_called_once()
            call_kwargs = mock_process.call_args[1]
            assert call_kwargs["summarization_strategy"] == "hierarchical"

    @pytest.mark.asyncio
    async def test_generate_document_summaries_with_shallow(self):
        """Test document summaries generation with shallow flag"""
        results = [
            [
                {
                    "metadata": {
                        "source_metadata": {"source_id": "/path/to/file1.pdf"},
                    },
                },
            ],
        ]

        with (
            patch(
                "nvidia_rag.utils.summarization.get_summarization_semaphore"
            ) as mock_semaphore,
            patch(
                "nvidia_rag.utils.summarization._process_single_file_summary"
            ) as mock_process,
            patch("nvidia_rag.utils.summarization.SUMMARY_STATUS_HANDLER"),
        ):
            mock_semaphore.return_value = asyncio.Semaphore(10)
            mock_process.return_value = {
                "file_name": "file1.pdf",
                "status": "SUCCESS",
            }

            stats = await generate_document_summaries(
                results=results,
                collection_name="test_collection",
                is_shallow=True,
            )

            assert stats["total_files"] == 1
            mock_process.assert_called_once()
            call_kwargs = mock_process.call_args[1]
            assert call_kwargs["is_shallow"] is True

    @pytest.mark.asyncio
    async def test_generate_document_summaries_with_custom_params(self):
        """Test document summaries generation with custom config and prompts"""
        results = [
            [
                {
                    "metadata": {
                        "source_metadata": {"source_id": "/path/to/file1.pdf"},
                    },
                },
            ],
        ]

        custom_config = NvidiaRAGConfig()
        custom_prompts = {
            "document_summary_prompt": {
                "system": "Test system",
                "human": "Test human",
            },
            "iterative_summary_prompt": {
                "system": "Test system",
                "human": "Test human",
            },
        }

        with (
            patch(
                "nvidia_rag.utils.summarization.get_summarization_semaphore"
            ) as mock_semaphore,
            patch(
                "nvidia_rag.utils.summarization._process_single_file_summary"
            ) as mock_process,
            patch("nvidia_rag.utils.summarization.SUMMARY_STATUS_HANDLER"),
        ):
            mock_semaphore.return_value = asyncio.Semaphore(10)
            mock_process.return_value = {
                "file_name": "file1.pdf",
                "status": "SUCCESS",
            }

            stats = await generate_document_summaries(
                results=results,
                collection_name="test_collection",
                config=custom_config,
                prompts=custom_prompts,
            )

            assert stats["total_files"] == 1
            assert stats["successful"] == 1
            mock_process.assert_called_once()
            call_kwargs = mock_process.call_args[1]
            assert call_kwargs["prompts"] == custom_prompts

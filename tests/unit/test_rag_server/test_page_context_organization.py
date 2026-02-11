# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for page context organization in RAG server."""

from unittest.mock import MagicMock

from langchain_core.documents import Document

from nvidia_rag.rag_server.main import NvidiaRAG


class TestFormatContextByPage:
    """Test cases for _format_context_by_page."""

    def test_format_context_by_page_with_page_numbers(self):
        """Test grouping docs by page with markers."""
        rag = NvidiaRAG()
        docs = [
            Document(
                page_content="Content from page 3",
                metadata={
                    "source": {"source_name": "/path/to/doc.pdf"},
                    "content_metadata": {"page_number": 3},
                },
            ),
            Document(
                page_content="Content from page 5",
                metadata={
                    "source": {"source_name": "/path/to/doc.pdf"},
                    "content_metadata": {"page_number": 5},
                },
            ),
        ]

        def format_fn(d):
            return f"Content: {d.page_content}"

        result = rag._format_context_by_page(docs, format_fn)

        assert "=== Page 3 (doc) ===" in result
        assert "=== Page 5 (doc) ===" in result
        assert "Content from page 3" in result
        assert "Content from page 5" in result
        assert result.index("Page 3") < result.index("Page 5")

    def test_format_context_by_page_with_no_page_numbers(self):
        """Test docs without page_number go to Additional context."""
        rag = NvidiaRAG()
        docs = [
            Document(
                page_content="No page info",
                metadata={"source": {"source_name": "/path/to/doc.txt"}},
            ),
        ]

        def format_fn(d):
            return d.page_content

        result = rag._format_context_by_page(docs, format_fn)

        assert "=== Additional context ===" in result
        assert "No page info" in result


class TestExtractPageSet:
    """Test cases for _extract_page_set_from_docs."""

    def test_extract_page_set_basic(self):
        """Test extracting page set from docs."""
        rag = NvidiaRAG()
        docs = [
            Document(
                page_content="x",
                metadata={
                    "source": {"source_name": "/path/to/a.pdf"},
                    "content_metadata": {"page_number": 3},
                    "collection_name": "col1",
                },
            ),
            Document(
                page_content="y",
                metadata={
                    "source": {"source_name": "/path/to/a.pdf"},
                    "content_metadata": {"page_number": 5},
                    "collection_name": "col1",
                },
            ),
        ]

        page_set = rag._extract_page_set_from_docs(docs)

        assert ("col1", "/path/to/a.pdf", 3) in page_set
        assert ("col1", "/path/to/a.pdf", 5) in page_set
        assert len(page_set) == 2

    def test_extract_page_set_skips_no_page_number(self):
        """Test docs without page_number are skipped."""
        rag = NvidiaRAG()
        docs = [
            Document(
                page_content="x",
                metadata={"source": {"source_name": "/path/to/a.pdf"}},
            ),
        ]

        page_set = rag._extract_page_set_from_docs(docs)

        assert len(page_set) == 0


class TestExpandPageSetWithNeighbors:
    """Test cases for _expand_page_set_with_neighbors."""

    def test_expand_with_one_neighbor(self):
        """Test expanding with 1 page before and after."""
        rag = NvidiaRAG()
        page_set = {("col1", "doc.pdf", 3), ("col1", "doc.pdf", 5)}

        expanded = rag._expand_page_set_with_neighbors(page_set, n=1)

        assert ("col1", "doc.pdf", 2) in expanded
        assert ("col1", "doc.pdf", 3) in expanded
        assert ("col1", "doc.pdf", 4) in expanded
        assert ("col1", "doc.pdf", 5) in expanded
        assert ("col1", "doc.pdf", 6) in expanded

    def test_expand_excludes_page_zero(self):
        """Test that page 0 is not added when first page (1) is retrieved."""
        rag = NvidiaRAG()
        page_set = {("col1", "doc.pdf", 1)}

        expanded = rag._expand_page_set_with_neighbors(page_set, n=1)

        assert ("col1", "doc.pdf", 0) not in expanded
        assert ("col1", "doc.pdf", 1) in expanded
        assert ("col1", "doc.pdf", 2) in expanded

    def test_expand_last_page_adds_page_plus_one(self):
        """Last page + 1 is added; VDB returns empty for non-existent pages (safe)."""
        rag = NvidiaRAG()
        page_set = {("col1", "doc.pdf", 10)}  # Assume last page

        expanded = rag._expand_page_set_with_neighbors(page_set, n=1)

        assert ("col1", "doc.pdf", 9) in expanded
        assert ("col1", "doc.pdf", 10) in expanded
        assert ("col1", "doc.pdf", 11) in expanded  # May not exist; fetch returns empty


class TestExpandAndOrganizeContext:
    """Test cases for _expand_and_organize_context."""

    def test_no_page_set_returns_docs_unchanged(self):
        """Text files with no page_number: return reranker top-k unchanged."""
        rag = NvidiaRAG()
        docs = [
            Document(
                page_content="x",
                metadata={"source": {"source_name": "/path/to/a.txt"}},
            ),
        ]
        mock_vdb = MagicMock()

        result = rag._expand_and_organize_context(
            docs=docs,
            vdb_op=mock_vdb,
            collection_names=["col1"],
            fetch_full_page_context=False,
            fetch_neighboring_pages=0,
        )

        assert result == docs
        mock_vdb.retrieve_chunks_by_filter.assert_not_called()

    def test_no_page_set_with_fetch_flags_returns_docs_unchanged(self):
        """Text files: even with fetch_full_page_context=True, return unchanged."""
        rag = NvidiaRAG()
        docs = [
            Document(
                page_content="plain text content",
                metadata={"source": {"source_name": "/path/to/notes.txt"}},
            ),
        ]
        mock_vdb = MagicMock()

        result = rag._expand_and_organize_context(
            docs=docs,
            vdb_op=mock_vdb,
            collection_names=["col1"],
            fetch_full_page_context=True,
            fetch_neighboring_pages=1,
        )

        assert result == docs
        mock_vdb.retrieve_chunks_by_filter.assert_not_called()

    def test_fetch_full_page_context_calls_vdb(self):
        """Test fetch_full_page_context triggers VDB filter query."""
        rag = NvidiaRAG()
        docs = [
            Document(
                page_content="x",
                metadata={
                    "source": {"source_name": "/path/to/a.pdf"},
                    "content_metadata": {"page_number": 3},
                    "collection_name": "col1",
                },
            ),
        ]
        mock_vdb = MagicMock()
        mock_vdb.retrieve_chunks_by_filter.return_value = []

        result = rag._expand_and_organize_context(
            docs=docs,
            vdb_op=mock_vdb,
            collection_names=["col1"],
            fetch_full_page_context=True,
            fetch_neighboring_pages=0,
        )

        mock_vdb.retrieve_chunks_by_filter.assert_called_once()
        call_kw = mock_vdb.retrieve_chunks_by_filter.call_args[1]
        assert call_kw["collection_name"] == "col1"
        assert call_kw["source_name"] == "/path/to/a.pdf"
        assert 3 in call_kw["page_numbers"]
        assert len(result) >= 1

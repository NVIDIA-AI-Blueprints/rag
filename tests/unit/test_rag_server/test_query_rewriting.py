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

"""
Test suite for query rewriting functionality in the RAG server.
"""

import pytest


class DummyPrompt:
    """A minimal LCEL-like object that supports piping and invoke/ainvoke/format_messages."""

    def __init__(self, rewritten_prefix: str = "REWRITTEN"):
        self.rewritten_prefix = rewritten_prefix

    def __or__(self, other):  # support chaining: prompt | llm | parser | output
        return self

    def invoke(self, inputs, config=None):
        # Mimic rewriter returning a transformed query from the provided input
        value = inputs.get("input") or inputs.get("question") or ""
        return f"{self.rewritten_prefix}({value})"

    async def ainvoke(self, inputs, config=None):
        # Async version of invoke
        value = inputs.get("input") or inputs.get("question") or ""
        return f"{self.rewritten_prefix}({value})"

    def stream(self, inputs, config=None):
        # Minimal streaming generator to satisfy generate() call path
        yield "ok"

    async def astream(self, inputs, config=None):
        # Minimal async streaming generator
        yield "ok"

    def format_messages(self, **kwargs):
        # Return a list-like structure for logging compatibility
        class Msg:
            def __init__(self, type_, content):
                self.type = type_
                self.content = content

        return [
            Msg("system", "dummy-system"),
            Msg("human", f"{kwargs}"),
        ]


class DummyVDB:
    """A minimal VDB stub used via monkeypatch on __prepare_vdb_op."""

    last_query = None

    def check_collection_exists(self, collection_name: str) -> bool:
        return True

    def get_langchain_vectorstore(self, collection_name: str):
        return object()

    def get_metadata_schema(self, collection_name: str):
        return []

    def retrieval_langchain(self, query, collection_name, vectorstore=None, top_k=None, filter_expr="", otel_ctx=None):
        """Sync method - called in ThreadPoolExecutor or directly."""
        DummyVDB.last_query = query
        return []


@pytest.fixture(autouse=True)
def stub_chat_prompt(monkeypatch):
    # Disable reflection for these tests so search/generate follow the
    # non-reflection code paths (reflection behaviour is covered elsewhere).
    monkeypatch.setenv("ENABLE_REFLECTION", "false")

    # Replace ChatPromptTemplate.from_messages to avoid real LCEL graph
    import nvidia_rag.rag_server.main as main

    class DummyChatPromptTemplate:
        @staticmethod
        def from_messages(messages):
            return DummyPrompt()

    monkeypatch.setattr(main, "ChatPromptTemplate", DummyChatPromptTemplate)

    # Ensure StreamingFilterThinkParser and StrOutputParser are no-ops in the chain
    class NoOpParser:
        def __ror__(self, other):
            return other

    # StreamingFilterThinkParser is now an instance attribute, not module-level
    # monkeypatch.setattr(main, "StreamingFilterThinkParser", NoOpParser())

    class NoOpStrOutputParser:
        def __ror__(self, other):
            return other

    monkeypatch.setattr(main, "StrOutputParser", lambda: NoOpStrOutputParser())

    # Stub LLM and ranker to avoid external calls during generate()
    monkeypatch.setattr(main, "get_llm", lambda **kwargs: DummyPrompt())
    monkeypatch.setattr(main, "get_ranking_model", lambda **kwargs: None)

    # Mock generate_answer_async to return a simple async generator
    async def mock_generate_answer_async(generator, contexts, **kw):
        yield "ok"

    monkeypatch.setattr(main, "generate_answer_async", mock_generate_answer_async)


@pytest.mark.asyncio
async def test_search_uses_query_rewriter_when_enabled(monkeypatch):
    """Test that query rewriting is used when enabled and messages are provided."""
    from nvidia_rag.rag_server.main import NvidiaRAG

    # Set CONVERSATION_HISTORY > 0, required for query rewriting to work
    monkeypatch.setenv("CONVERSATION_HISTORY", "5")

    fake_vdb = DummyVDB()
    rag = NvidiaRAG()
    # Force using our stubbed vdb_op inside generate/search path that may call __prepare_vdb_op
    monkeypatch.setattr(NvidiaRAG, "_prepare_vdb_op", lambda self, **kw: fake_vdb)

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is RAG?"},
        {"role": "assistant", "content": "A retrieval-augmented framework."},
    ]

    # Act
    await rag.search(
        query="How does it work?",
        messages=messages,
        collection_names=["test"],
        enable_query_rewriting=True,
        enable_reranker=False,
        filter_expr="",
    )

    # Assert: rewritten query should be used for retrieval
    assert fake_vdb.last_query == "REWRITTEN(How does it work?)"


@pytest.mark.asyncio
async def test_search_skips_query_rewriter_when_history_is_zero(monkeypatch, caplog):
    """Test that query rewriting is skipped with a warning when CONVERSATION_HISTORY=0."""
    from nvidia_rag.rag_server.main import NvidiaRAG
    import logging

    # Set CONVERSATION_HISTORY to 0 (default)
    monkeypatch.setenv("CONVERSATION_HISTORY", "0")

    fake_vdb = DummyVDB()
    rag = NvidiaRAG()
    # Force using our stubbed vdb_op inside generate/search path that may call __prepare_vdb_op
    monkeypatch.setattr(NvidiaRAG, "_prepare_vdb_op", lambda self, **kw: fake_vdb)

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is RAG?"},
        {"role": "assistant", "content": "A retrieval-augmented framework."},
    ]

    # Act
    with caplog.at_level(logging.WARNING):
        await rag.search(
            query="How does it work?",
            messages=messages,
            collection_names=["test"],
            enable_query_rewriting=True,
            enable_reranker=False,
            filter_expr="",
        )

    # Assert: query rewriting should be skipped and original query used
    assert fake_vdb.last_query == "How does it work?"
    # Assert: a warning should be logged
    assert any(
        "Query rewriting is enabled but CONVERSATION_HISTORY is set to 0" in record.message
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_search_uses_only_current_query_when_history_disabled(monkeypatch):
    """Test that when multiturn_retrieval_simple is False (default), only current query is used."""
    from nvidia_rag.rag_server.main import NvidiaRAG

    fake_vdb = DummyVDB()
    rag = NvidiaRAG()
    monkeypatch.setattr(NvidiaRAG, "_prepare_vdb_op", lambda self, **kw: fake_vdb)

    messages = [
        {"role": "user", "content": "What is RAG?"},
        {"role": "assistant", "content": "A retrieval-augmented framework."},
    ]

    # Act: multiturn_retrieval_simple defaults to False
    await rag.search(
        query="How does it work?",
        messages=messages,
        collection_names=["test"],
        enable_query_rewriting=False,
        enable_reranker=False,
        filter_expr="",
    )

    # Assert: only current query is used (no history concatenation)
    assert fake_vdb.last_query == "How does it work?"


@pytest.mark.asyncio
async def test_search_combines_history_when_multiturn_enabled(monkeypatch):
    """Test that when multiturn_retrieval_simple is True, history is concatenated."""
    from nvidia_rag.rag_server.main import NvidiaRAG

    # Enable multiturn retrieval via environment variable BEFORE creating NvidiaRAG instance
    monkeypatch.setenv("MULTITURN_RETRIEVER_SIMPLE", "True")
    
    fake_vdb = DummyVDB()
    rag = NvidiaRAG()
    monkeypatch.setattr(NvidiaRAG, "_prepare_vdb_op", lambda self, **kw: fake_vdb)

    messages = [
        {"role": "user", "content": "What is RAG?"},
        {"role": "assistant", "content": "A retrieval-augmented framework."},
    ]

    # Act
    await rag.search(
        query="How does it work?",
        messages=messages,
        collection_names=["test"],
        enable_query_rewriting=False,
        enable_reranker=False,
        filter_expr="",
    )

    # Assert: history + current query should be concatenated with '. '
    assert fake_vdb.last_query == "What is RAG?. How does it work?"


@pytest.mark.asyncio
async def test_generate_uses_query_rewriter_when_enabled(monkeypatch):
    """Test that query rewriting is used in generate when enabled with conversation history."""
    from nvidia_rag.rag_server.main import NvidiaRAG

    fake_vdb = DummyVDB()
    # Set CONVERSATION_HISTORY > 0 so chat_history is not empty (query rewriting requires chat history)
    monkeypatch.setenv("CONVERSATION_HISTORY", "5")
    # Ensure multiturn simple retrieval is disabled (test relies on query rewriting)
    monkeypatch.setenv("MULTITURN_RETRIEVER_SIMPLE", "False")
    rag = NvidiaRAG()
    monkeypatch.setattr(NvidiaRAG, "_prepare_vdb_op", lambda self, **kw: fake_vdb)

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is RAG?"},
        {"role": "assistant", "content": "A retrieval-augmented framework."},
        {"role": "user", "content": "How does it work?"},
    ]

    # Act: Calling generate() triggers retrieval before returning the stream
    stream = await rag.generate(
        messages=messages,
        use_knowledge_base=True,
        collection_names=["test"],
        enable_query_rewriting=True,
        enable_reranker=False,
        enable_vlm_inference=False,
        filter_expr="",
    )

    # Assert: rewritten query is used for retrieval inside RAG flow
    assert fake_vdb.last_query == "REWRITTEN(How does it work?)"


@pytest.mark.asyncio
async def test_generate_uses_only_current_query_when_history_disabled(monkeypatch):
    """Test that when multiturn_retrieval_simple is False (default), only current query is used."""
    from nvidia_rag.rag_server.main import NvidiaRAG

    fake_vdb = DummyVDB()
    # Explicitly ensure multiturn simple retrieval is disabled
    monkeypatch.setenv("MULTITURN_RETRIEVER_SIMPLE", "False")
    rag = NvidiaRAG()
    monkeypatch.setattr(NvidiaRAG, "_prepare_vdb_op", lambda self, **kw: fake_vdb)

    messages = [
        {"role": "user", "content": "What is RAG?"},
        {"role": "assistant", "content": "A retrieval-augmented framework."},
        {"role": "user", "content": "How does it work?"},
    ]

    stream = await rag.generate(
        messages=messages,
        use_knowledge_base=True,
        collection_names=["test"],
        enable_query_rewriting=False,
        enable_reranker=False,
        enable_vlm_inference=False,
        filter_expr="",
    )

    # Assert: only current query is used (no history concatenation)
    assert fake_vdb.last_query == "How does it work?"


@pytest.mark.asyncio
async def test_generate_combines_history_when_multiturn_enabled(monkeypatch):
    """Test that when multiturn_retrieval_simple is True, history is concatenated."""
    from nvidia_rag.rag_server.main import NvidiaRAG
    
    # Enable multiturn retrieval via environment variable BEFORE creating NvidiaRAG instance
    # Also set CONVERSATION_HISTORY > 0 so chat_history is not empty
    monkeypatch.setenv("MULTITURN_RETRIEVER_SIMPLE", "True")
    monkeypatch.setenv("CONVERSATION_HISTORY", "5")
    
    fake_vdb = DummyVDB()
    rag = NvidiaRAG()
    monkeypatch.setattr(NvidiaRAG, "_prepare_vdb_op", lambda self, **kw: fake_vdb)

    messages = [
        {"role": "user", "content": "What is RAG?"},
        {"role": "assistant", "content": "A retrieval-augmented framework."},
        {"role": "user", "content": "How does it work?"},
    ]

    stream = await rag.generate(
        messages=messages,
        use_knowledge_base=True,
        collection_names=["test"],
        enable_query_rewriting=False,
        enable_reranker=False,
        enable_vlm_inference=False,
        filter_expr="",
    )

    # In _rag_chain when multiturn_retrieval_simple is enabled, 
    # last previous user query is combined with current retriever_query
    # Expected concatenation: "What is RAG?. How does it work?"
    assert fake_vdb.last_query == "What is RAG?. How does it work?"


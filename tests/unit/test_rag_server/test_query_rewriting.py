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
    """A minimal LCEL-like object that supports piping and invoke/format_messages."""

    def __init__(self, rewritten_prefix: str = "REWRITTEN"):
        self.rewritten_prefix = rewritten_prefix

    def __or__(self, other):  # support chaining: prompt | llm | parser | output
        return self

    def invoke(self, inputs, config=None):
        # Mimic rewriter returning a transformed query from the provided input
        value = inputs.get("input") or inputs.get("question") or ""
        return f"{self.rewritten_prefix}({value})"

    def stream(self, inputs, config=None):
        # Minimal streaming generator to satisfy generate() call path
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
        self.last_query = query
        return []


@pytest.fixture(autouse=True)
def stub_chat_prompt(monkeypatch):
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

    monkeypatch.setattr(main, "StreamingFilterThinkParser", NoOpParser())

    class NoOpStrOutputParser:
        def __ror__(self, other):
            return other

    monkeypatch.setattr(main, "StrOutputParser", lambda: NoOpStrOutputParser())

    # Stub LLM and ranker to avoid external calls during generate()
    monkeypatch.setattr(main, "get_llm", lambda **kwargs: DummyPrompt())
    monkeypatch.setattr(main, "get_ranking_model", lambda **kwargs: None)
    monkeypatch.setattr(main, "query_rewriter_llm", DummyPrompt())
    # Make generate() return a simple sync iterator instead of async coroutine
    monkeypatch.setattr(
        main,
        "generate_answer",
        lambda generator, contexts, **kw: iter(["ok"])  # return sync iterator
    )


def test_search_uses_query_rewriter_when_enabled(monkeypatch):
    from nvidia_rag.rag_server.main import NvidiaRAG

    fake_vdb = DummyVDB()
    rag = NvidiaRAG()
    # Force using our stubbed vdb_op inside generate/search path that may call __prepare_vdb_op
    monkeypatch.setattr(NvidiaRAG, "_NvidiaRAG__prepare_vdb_op", lambda self, **kw: fake_vdb)

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is RAG?"},
        {"role": "assistant", "content": "A retrieval-augmented framework."},
    ]

    # Act
    rag.search(
        query="How does it work?",
        messages=messages,
        collection_name="test",
        enable_query_rewriting=True,
        enable_reranker=False,
        filter_expr="",
    )

    # Assert: rewritten query should be used for retrieval
    assert fake_vdb.last_query == "REWRITTEN(How does it work?)"


def test_search_combines_history_when_rewriter_disabled(monkeypatch):
    from nvidia_rag.rag_server.main import NvidiaRAG

    fake_vdb = DummyVDB()
    rag = NvidiaRAG()
    monkeypatch.setattr(NvidiaRAG, "_NvidiaRAG__prepare_vdb_op", lambda self, **kw: fake_vdb)

    messages = [
        {"role": "user", "content": "What is RAG?"},
        {"role": "assistant", "content": "A retrieval-augmented framework."},
    ]

    # Act
    rag.search(
        query="How does it work?",
        messages=messages,
        collection_name="test",
        enable_query_rewriting=False,
        enable_reranker=False,
        filter_expr="",
    )

    # Assert: history + current query should be concatenated with '. '
    assert fake_vdb.last_query == "What is RAG?. How does it work?"


def test_generate_uses_query_rewriter_when_enabled(monkeypatch):
    # We validate that when use_knowledge_base=True, generate() calls the RAG chain
    # and passes the rewritten query into retrieval via FakeVDB
    from nvidia_rag.rag_server.main import NvidiaRAG

    fake_vdb = DummyVDB()
    rag = NvidiaRAG()
    monkeypatch.setattr(NvidiaRAG, "_NvidiaRAG__prepare_vdb_op", lambda self, **kw: fake_vdb)

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is RAG?"},
        {"role": "assistant", "content": "A retrieval-augmented framework."},
        {"role": "user", "content": "How does it work?"},
    ]

    # Act: Calling generate() triggers retrieval before returning the stream
    stream = rag.generate(
        messages=messages,
        use_knowledge_base=True,
        collection_name="test",
        enable_query_rewriting=True,
        enable_reranker=False,
        enable_vlm_inference=False,
        filter_expr="",
    )

    # Assert: rewritten query is used for retrieval inside RAG flow
    assert fake_vdb.last_query == "REWRITTEN(How does it work?)"


def test_generate_combines_history_when_rewriter_disabled(monkeypatch):
    # When query rewriting is disabled, generate() should concatenate prior user query
    from nvidia_rag.rag_server.main import NvidiaRAG

    fake_vdb = DummyVDB()
    rag = NvidiaRAG()
    monkeypatch.setattr(NvidiaRAG, "_NvidiaRAG__prepare_vdb_op", lambda self, **kw: fake_vdb)

    messages = [
        {"role": "user", "content": "What is RAG?"},
        {"role": "assistant", "content": "A retrieval-augmented framework."},
        {"role": "user", "content": "How does it work?"},
    ]

    stream = rag.generate(
        messages=messages,
        use_knowledge_base=True,
        collection_name="test",
        enable_query_rewriting=False,
        enable_reranker=False,
        enable_vlm_inference=False,
        filter_expr="",
    )

    # In __rag_chain when disabled, only last user query is combined with most recent prior user query
    # The combination logic for generate path uses only the last user message from chat_history list slice [-1:]
    # But here chat_history contains two user messages; __rag_chain combines last previous user query and current retriever_query
    # For generate(), retriever_query is built from the last user message content only
    # So when disabled, it will combine previous user message and current one.
    # Expected concatenation: "What is RAG?. How does it work?"
    assert fake_vdb.last_query == "What is RAG?. How does it work?"


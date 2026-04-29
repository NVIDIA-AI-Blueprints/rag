# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Agentic RAG package for nvidia_rag.rag_server.

Public API:

  AgenticRag              — LangGraph plan-and-execute agent (core logic).
  AgenticRAGGraphState    — Pydantic state model for the LangGraph graph.
  AgenticSearchParams     — Dataclass of all per-request NvidiaRAG.search()
                            parameters forwarded by the agentic retriever.
  _agentic_search_params  — ContextVar[AgenticSearchParams]; set by
                            NvidiaRAG._agentic_chain before graph.ainvoke().
  build_agentic_rag_agent — Factory: builds + compiles the agent from a
                            NvidiaRAG instance and its AgenticRAGConfig.
  make_retriever_fn       — Creates an async retriever backed by NvidiaRAG.search().
"""

from nvidia_rag.rag_server.agentic_rag.agentic_rag import (
    AgenticRag,
    AgenticRAGGraphState,
)
from nvidia_rag.rag_server.agentic_rag.builder import (
    AgenticSearchParams,
    _agentic_search_params,
    build_agentic_rag_agent,
    make_retriever_fn,
)
from nvidia_rag.rag_server.agentic_rag.runner import run_agentic_pipeline

__all__ = [
    "AgenticRag",
    "AgenticRAGGraphState",
    "AgenticSearchParams",
    "_agentic_search_params",
    "build_agentic_rag_agent",
    "make_retriever_fn",
    "run_agentic_pipeline",
]

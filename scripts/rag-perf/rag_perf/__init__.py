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
rag-perf — RAG Server Performance Benchmarking Tool.

The package is organised by direction of data flow:

  config     Pydantic v2 RunConfig + nested sub-configs + the three enums
             (InputSource / SyntheticMode / LoadMode).
  query      Query loading and synthetic LLM-based query generation.
             Owns SyntheticQueryGenerator and QueryLoader.
  reporting  Output side: shared dataclasses (ProfileRecord, ProfileResult,
             StageBreakdown, CitationQuality, RagMetricsSummary), plus the
             MetricsAggregator and Reporter service classes.
  runner     Execution side: RagProfiler (async httpx + SSE parsing),
             AiperfRunner (subprocess wrapper), BenchmarkRunner (the
             top-level run() orchestrator that handles single-point and
             grid-sweep modes uniformly).
  cli        Single Click command for ``rag-perf -c <config>``, plus the
             startup banner.
  plugin     aiperf endpoint plugin (``nvidia_rag``); registered via the
             ``aiperf.plugins`` entry point in pyproject.toml.

The names below are re-exported here for convenience; user code should
prefer ``from rag_perf import RunConfig, BenchmarkRunner, …``.
"""

__version__ = "0.1.0"

from rag_perf.cli import print_banner
from rag_perf.config import (
    AiperfConfig,
    GenerationParams,
    InputConfig,
    InputSource,
    LoadConfig,
    LoadMode,
    OutputConfig,
    RagParams,
    RunConfig,
    SyntheticInputConfig,
    SyntheticMode,
    TargetConfig,
)
from rag_perf.query import PromptTemplates, QueryLoader, SyntheticQueryGenerator
from rag_perf.reporting import (
    CitationQuality,
    MetricsAggregator,
    ProfileRecord,
    ProfileResult,
    RagMetricsSummary,
    Reporter,
    StageBreakdown,
)
from rag_perf.runner import AiperfRunner, BenchmarkRunner, RagProfiler

__all__ = [
    "__version__",
    # Service classes
    "AiperfRunner",
    "BenchmarkRunner",
    "MetricsAggregator",
    "QueryLoader",
    "RagProfiler",
    "Reporter",
    "SyntheticQueryGenerator",
    # Configuration models
    "AiperfConfig",
    "GenerationParams",
    "InputConfig",
    "InputSource",
    "LoadConfig",
    "LoadMode",
    "OutputConfig",
    "RagParams",
    "RunConfig",
    "SyntheticInputConfig",
    "SyntheticMode",
    "TargetConfig",
    # Data models
    "CitationQuality",
    "ProfileRecord",
    "ProfileResult",
    "PromptTemplates",
    "RagMetricsSummary",
    "StageBreakdown",
    # Utility
    "print_banner",
]

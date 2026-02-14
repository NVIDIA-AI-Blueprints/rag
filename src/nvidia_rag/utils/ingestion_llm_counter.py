# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Global counter for summary LLM calls during ingestion.

Used to report how many LLM invocations occurred for summarization in a single
ingestion run. Only summary-related LLM calls are counted (e.g. RAPTOR cluster
summaries, single/iterative/hierarchical document summaries). Other ingestor
logic does not use LLM; this counter is reset at the start of each ingestion
and exposed in the final ingestion response as summary_llm_calls.
"""

import threading

_lock = threading.Lock()
_count: int = 0


def increment_summary_llm_count() -> None:
    """Increment the global summary LLM call count by one. Thread-safe."""
    global _count
    with _lock:
        _count += 1


def get_summary_llm_count() -> int:
    """Return the current summary LLM call count. Thread-safe."""
    with _lock:
        return _count


def reset_summary_llm_count() -> None:
    """Reset the counter to zero. Call at the start of an ingestion run. Thread-safe."""
    global _count
    with _lock:
        _count = 0

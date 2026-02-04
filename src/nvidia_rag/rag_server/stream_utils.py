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

"""Stream utilities for async stream processing."""

import logging
from typing import Any
from collections.abc import AsyncGenerator

logger = logging.getLogger(__name__)


async def _async_iter(items) -> AsyncGenerator[Any, None]:
    """Helper to convert a list to an async generator."""
    for item in items:
        yield item


async def _eager_prefetch_astream(stream_gen):
    """
    Eagerly fetch the first chunk from an async stream to trigger any errors early.

    Args:
        stream_gen: Async generator to prefetch from

    Returns:
        Async generator that yields the prefetched chunk followed by the rest

    Raises:
        StopAsyncIteration: If the stream is empty (converted to empty generator)
    """
    try:
        first_chunk = await stream_gen.__anext__()

        async def complete_stream():
            yield first_chunk
            async for chunk in stream_gen:
                yield chunk

        return complete_stream()
    except StopAsyncIteration:
        logger.warning("LLM produced no output.")

        async def empty_gen():
            return
            yield  # Make it an async generator

        return empty_gen()

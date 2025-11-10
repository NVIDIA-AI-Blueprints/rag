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

"""The wrapper for interacting with reranking models.
1. _get_ranking_model: Creates the ranking model instance.
2. get_ranking_model: Returns the ranking model instance if it doesn't exist in cache.
"""

import logging
from functools import lru_cache

from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_nvidia_ai_endpoints import NVIDIARerank

from nvidia_rag.utils.common import sanitize_nim_url
from nvidia_rag.utils.configuration import NvidiaRAGConfig

logger = logging.getLogger(__name__)


@lru_cache
def _get_ranking_model(model="", url="", top_n=4, config: NvidiaRAGConfig | None = None) -> BaseDocumentCompressor:
    """Create the ranking model.
    
    Args:
        model: Model name
        url: URL endpoint
        top_n: Number of top results
        config: NvidiaRAGConfig instance. If None, creates a new one.

    Returns:
        BaseDocumentCompressor: Base class for document compressors.

    Raises:
        RuntimeError: If the ranking model engine is not supported or initialization fails.
    """
    if config is None:
        config = NvidiaRAGConfig()

    # Sanitize the URL
    url = sanitize_nim_url(url, model, "ranking")

    # Validate top_n
    if top_n is None or top_n <= 0:
        logger.warning("top_n must be a positive integer, setting to 4")
        top_n = 4

    if config.ranking.model_engine == "nvidia-ai-endpoints":
        if url:
            logger.info("Using ranking model hosted at %s", url)
            return NVIDIARerank(base_url=url, top_n=top_n, truncate="END")

        if model:
            logger.info("Using ranking model %s hosted at api catalog", model)
            return NVIDIARerank(model=model, top_n=top_n, truncate="END")

        # No model or URL provided
        raise RuntimeError(
            f"Ranking model configuration incomplete. "
            f"Either 'model' or 'url' must be provided. "
            f"Received: model='{model}', url='{url}'"
        )

    # Unsupported engine
    raise RuntimeError(
        f"Unsupported ranking model engine: '{config.ranking.model_engine}'. "
        f"Supported engines: 'nvidia-ai-endpoints'"
    )


def get_ranking_model(model="", url="", top_n=4, config: NvidiaRAGConfig | None = None) -> BaseDocumentCompressor:
    """Create the ranking model.
    
    Args:
        model: Model name
        url: URL endpoint
        top_n: Number of top results
        config: NvidiaRAGConfig instance. If None, creates a new one.

    Returns:
        BaseDocumentCompressor: The ranking model instance.

    Raises:
        RuntimeError: If the ranking model cannot be created.
    """
    if config is None:
        config = NvidiaRAGConfig()
    return _get_ranking_model(model, url, top_n, config)

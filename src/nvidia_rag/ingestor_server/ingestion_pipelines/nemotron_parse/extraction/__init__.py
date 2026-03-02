# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Nemotron Parse extraction pipeline (vendored for ingestor-server).

PDF → render → parse → optional VLM caption → result list. Supports sequential or Ray-based parallel page processing.
"""

from nvidia_rag.ingestor_server.ingestion_pipelines.nemotron_parse.extraction.config import (
    build_extraction_config,
)
from nvidia_rag.ingestor_server.ingestion_pipelines.nemotron_parse.extraction.pipeline import (
    ExtractionPipeline,
)

__all__ = [
    "build_extraction_config",
    "ExtractionPipeline",
]

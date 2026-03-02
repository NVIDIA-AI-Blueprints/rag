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
This module contains Nemotron parse based ingestion pipeline.
"""
from typing import Any
import asyncio
import logging

from nvidia_rag.utils.configuration import NvidiaRAGConfig
from nvidia_rag.utils.vdb.vdb_base import VDBRag
from nvidia_rag.ingestor_server.ingestion_state_manager import IngestionStateManager

from nvidia_rag.utils.observability.tracing import (
    get_tracer,
    trace_function,
)
from nvidia_rag.utils.embedding import get_embedding_model

from nvidia_rag.ingestor_server.ingestion_pipelines.helper import format_pipeline_config


logger = logging.getLogger(__name__)
TRACER = get_tracer("nvidia_rag.ingestor.nemotron_parse")

EMBED_CONCURRENCY = 4


class NemotronParse:
    """
    Nemotron parse based ingestion pipeline.
    """

    def __init__(
        self,

        # Configuration arguments
        config: NvidiaRAGConfig,
        vdb_op: VDBRag,
        state_manager: IngestionStateManager,

        # Pipeline arguments
        filepaths: list[str],
        collection_name: str,
        split_options: dict[str, Any] | None = None,
    ):
        """
        Initialize the NemotronParse class.
        """
        self.config = config
        self.vdb_op = vdb_op
        self.state_manager = state_manager

        # Pipeline arguments
        self.filepaths = filepaths
        self.collection_name = collection_name
        self.split_options = split_options

        # Initialize embedding model
        self.document_embedder = get_embedding_model(
                model=self.config.embeddings.model_name,
                url=self.config.embeddings.server_url,
                config=self.config,
            )

    @trace_function("ingestor.ingestion_pipelines.nemotron_parse.nemotron_parse_pipeline.run", tracer=TRACER)
    async def run(self):
        """
        Run the NemotronParse pipeline.
        """
        return await self.__run_nemotron_parse_pipeline()

    @trace_function("ingestor.ingestion_pipelines.nemotron_parse.nemotron_parse_pipeline.__run_nemotron_parse_pipeline", tracer=TRACER)
    async def __run_nemotron_parse_pipeline(self):
        """
        Run the NemotronParse pipeline.

        Performs following steps:
        - Perform extraction using Nemotron Parse
        - Embed result elements
        - Return results (main uploads to MinIO then calls vdb_op.run())

        Returns:
        - results: list[dict[str, Any]] - List of results
        - failures: list[dict[str, Any]] - List of failures
        """
        logger.info(
            "Running Nemotron Parse pipeline\n%s",
            format_pipeline_config("Nemotron Parse", self.config.nemotron_parse),
        )

        # Perform extraction using Nemotron Parse and add embedding to the results
        results, failures = await self.__perform_extraction_using_nemotron_parse()

        # Vectorstore insert is done in main after MinIO upload so nv_ingest can find image locations
        return results, failures

    @trace_function("ingestor.ingestion_pipelines.nemotron_parse.nemotron_parse_pipeline.__perform_extraction_using_nemotron_parse", tracer=TRACER)
    async def __perform_extraction_using_nemotron_parse(self):
        """
        Perform extraction using Nemotron Parse (extraction pipeline + adapter).
        Returns (results, failures) with results in ingestor schema; embeddings applied.
        """
        logger.info("Performing extraction using Nemotron Parse")

        from nvidia_rag.ingestor_server.ingestion_pipelines.nemotron_parse.extraction_adapter import (
            run_extraction,
        )
        from nvidia_rag.ingestor_server.ingestion_pipelines.nemotron_parse.nemotron_parse_embedding import (
            NemotronParseEmbedding,
        )

        results, failures = run_extraction(
            filepaths=self.filepaths,
            rag_config=self.config,
            collection_name=self.collection_name,
            max_pages_per_pdf=None,
        )

        nemotron_parse_embedding = NemotronParseEmbedding(self.config)
        for result in results:
            for el in result:
                nemotron_parse_embedding.embed_result_element(el)

        return results, failures
    

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
This is the module for NV-Ingest client wrapper.
1. Get NV-Ingest client: get_nv_ingest_client()
2. Get NV-Ingest ingestor: get_nv_ingest_ingestor()
"""

import logging
import os

from nv_ingest_client.client import Ingestor, NvIngestClient

from nvidia_rag.utils.common import (
    get_config,
    get_env_variable,
    sanitize_nim_url,
)
from nvidia_rag.utils.vdb.vdb_base import VDBRag

logger = logging.getLogger(__name__)

ENABLE_NV_INGEST_VDB_UPLOAD = (
    True  # When enabled entire ingestion would be performed using nv-ingest
)


def get_nv_ingest_client():
    """
    Creates and returns NV-Ingest client
    """
    config = get_config()

    client = NvIngestClient(
        # Host where nv-ingest-ms-runtime is running
        message_client_hostname=config.nv_ingest.message_client_hostname,
        # REST port, defaults to 7670
        message_client_port=config.nv_ingest.message_client_port,
    )
    return client


def get_nv_ingest_ingestor(
    nv_ingest_client_instance,
    filepaths: list[str],
    split_options=None,
    vdb_op: VDBRag = None,
):
    """
    Prepare NV-Ingest ingestor instance based on nv-ingest configuration

    Args:
        nv_ingest_client_instance: NvIngestClient instance
        filepaths: List of file paths to ingest
        split_options: Options for splitting documents

    Returns:
        - ingestor: Ingestor - NV-Ingest ingestor instance with configured tasks
    """
    config = get_config()

    logger.debug("Preparing NV Ingest Ingestor instance for filepaths: %s", filepaths)
    # Prepare the ingestor using nv-ingest-client
    ingestor = Ingestor(client=nv_ingest_client_instance)

    # Add files to ingestor
    ingestor = ingestor.files(filepaths)

    # Add extraction task
    # Determine table_output_format
    table_output_format = (
        "markdown" if config.nv_ingest.extract_tables else "pseudo_markdown"
    )
    # Create kwargs for extract method
    extract_kwargs = {
        "extract_text": config.nv_ingest.extract_text,
        "extract_infographics": config.nv_ingest.extract_infographics,
        "extract_tables": config.nv_ingest.extract_tables,
        "extract_charts": config.nv_ingest.extract_charts,
        "extract_images": config.nv_ingest.extract_images,
        "extract_method": config.nv_ingest.pdf_extract_method,
        "text_depth": config.nv_ingest.text_depth,
        "table_output_format": table_output_format,
        "extract_audio_params": {"segment_audio": config.nv_ingest.segment_audio},
        "extract_page_as_image": config.nv_ingest.extract_page_as_image,
    }
    if config.nv_ingest.pdf_extract_method in ["None", "none"]:
        extract_kwargs.pop("extract_method")
    else:
        logger.info(
            f"Extract method used for ingestion: {config.nv_ingest.pdf_extract_method}"
        )
    ingestor = ingestor.extract(**extract_kwargs)

    # Add splitting task (By default only works for text documents)
    split_options = split_options or {}
    split_source_types = ["text", "html", "mp3", "docx"]
    split_source_types = (
        ["PDF"] + split_source_types
        if config.nv_ingest.enable_pdf_splitter
        else split_source_types
    )
    logger.info(
        f"Post chunk split status: {config.nv_ingest.enable_pdf_splitter}. Splitting by: {split_source_types}"
    )
    ingestor = ingestor.split(
        tokenizer=config.nv_ingest.tokenizer,
        chunk_size=split_options.get("chunk_size", config.nv_ingest.chunk_size),
        chunk_overlap=split_options.get(
            "chunk_overlap", config.nv_ingest.chunk_overlap
        ),
        params={"split_source_types": split_source_types},
    )

    # Add captioning task if extract_images is enabled
    if config.nv_ingest.extract_images:
        logger.info(
            f"Enabling captioning task. Captioning Endpoint URL: {config.nv_ingest.caption_endpoint_url}, Captioning Model Name: {config.nv_ingest.caption_model_name}"
        )
        ingestor = ingestor.caption(
            api_key=get_env_variable(variable_name="NGC_API_KEY", default_value=""),
            endpoint_url=config.nv_ingest.caption_endpoint_url,
            model_name=config.nv_ingest.caption_model_name,
        )

    # Add Embedding task
    if ENABLE_NV_INGEST_VDB_UPLOAD:
        embedding_url = sanitize_nim_url(
            config.embeddings.server_url, config.embeddings.model_name, "embedding"
        )
        logger.info(
            f"Enabling embedding task. Embedding Endpoint URL: {embedding_url}, Embedding Model Name: {config.embeddings.model_name}"
        )
        if config.nv_ingest.structured_elements_modality:
            ingestor = ingestor.embed(
                structured_elements_modality=config.nv_ingest.structured_elements_modality,
                endpoint_url=embedding_url,
                model_name=config.embeddings.model_name,
            )
        elif config.nv_ingest.image_elements_modality:
            ingestor = ingestor.embed(
                image_elements_modality=config.nv_ingest.image_elements_modality,
                endpoint_url=embedding_url,
                model_name=config.embeddings.model_name,
            )
        else:
            ingestor = ingestor.embed(
                endpoint_url=embedding_url,
                model_name=config.embeddings.model_name,
            )

    # Add save to disk task
    if config.nv_ingest.save_to_disk:
        output_directory = os.path.join(
            os.getenv("INGESTOR_SERVER_DATA_DIR", "/data/"),
            "nv-ingest-results",
            vdb_op.collection_name,
        )
        os.makedirs(output_directory, exist_ok=True)
        ingestor = ingestor.save_to_disk(
            output_directory=output_directory,
            cleanup=not config.nv_ingest.save_to_disk,
        )

    # Add Vector-DB upload task
    if ENABLE_NV_INGEST_VDB_UPLOAD:
        ingestor = ingestor.vdb_upload(
            vdb_op=vdb_op,
            purge_results_after_upload=not config.nv_ingest.save_to_disk,
        )

    return ingestor

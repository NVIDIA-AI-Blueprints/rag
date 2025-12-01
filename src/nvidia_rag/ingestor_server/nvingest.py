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
import time

from nv_ingest_client.client import Ingestor, NvIngestClient

from nvidia_rag.utils.common import sanitize_nim_url
from nvidia_rag.utils.configuration import NvidiaRAGConfig
from nvidia_rag.utils.vdb.vdb_base import VDBRag

logger = logging.getLogger(__name__)


def _patched_wait_for_index(collection_name: str, num_elements: int, client):
    """
    Patched version of wait_for_index that fixes the race condition bug.
    
    The original function fails because it captures already_indexed_rows AFTER flush,
    by which time some/all new rows might already be indexed.
    
    This version:
    1. Captures the indexed row count BEFORE flush (proper baseline)
    2. Uses >= comparison to handle edge cases
    """
    from pymilvus import utility
    
    index_names = utility.list_indexes(collection_name)
    
    # FIXED: Capture the indexed row count BEFORE flush
    # This gives us the correct baseline before indexing starts
    pre_flush_counts = {}
    for index_name in index_names:
        pre_flush_counts[index_name] = client.describe_index(collection_name, index_name)["indexed_rows"]
        logger.info(
            f"Pre-flush indexed rows for {collection_name}, {index_name}: {pre_flush_counts[index_name]}"
        )
    
    # Now flush to trigger indexing
    client.flush(collection_name)
    
    indexed_rows = 0
    for index_name in index_names:
        indexed_rows = 0
        already_indexed_rows = pre_flush_counts[index_name]  # Use pre-flush count as baseline
        target_rows = already_indexed_rows + num_elements
        
        logger.info(
            f"Waiting for index: {collection_name}, {index_name} - "
            f"baseline: {already_indexed_rows}, inserting: {num_elements}, target: {target_rows}"
        )
        
        while indexed_rows < target_rows:
            pos_movement = 10  # number of iterations allowed without noticing an increase in indexed_rows
            for i in range(20):
                new_indexed_rows = client.describe_index(collection_name, index_name)["indexed_rows"]
                time.sleep(1)
                logger.info(
                    f"polling for indexed rows, {collection_name}, {index_name} - {new_indexed_rows} / {target_rows}"
                )
                # FIXED: Use >= instead of == to handle edge cases
                if new_indexed_rows >= target_rows:
                    indexed_rows = new_indexed_rows
                    logger.info(
                        f"Indexing complete for {collection_name}, {index_name}: {new_indexed_rows} rows indexed"
                    )
                    break
                # check if indexed_rows is staying the same, too many times means something is wrong
                if new_indexed_rows == indexed_rows:
                    pos_movement -= 1
                else:
                    pos_movement = 10
                # if pos_movement is 0, raise an error, means the rows are not getting indexed as expected
                if pos_movement == 0:
                    raise ValueError(
                        f"Rows are not getting indexed as expected. "
                        f"Current: {new_indexed_rows}, Target: {target_rows}, "
                        f"Pre-flush baseline: {already_indexed_rows}, "
                        f"Elements to insert: {num_elements}"
                    )
                indexed_rows = new_indexed_rows
    return indexed_rows


def _apply_wait_for_index_patch():
    """
    Apply monkey-patch to fix the wait_for_index race condition in nv_ingest_client.
    This should be called once before any ingestion operations.
    """
    try:
        import nv_ingest_client.util.vdb.milvus as milvus_module
        milvus_module.wait_for_index = _patched_wait_for_index
        logger.info("Successfully applied wait_for_index patch to nv_ingest_client")
    except ImportError:
        logger.warning("Could not import nv_ingest_client.util.vdb.milvus for patching")
    except Exception as e:
        logger.warning(f"Failed to apply wait_for_index patch: {e}")


# Apply the patch when this module is loaded
_apply_wait_for_index_patch()


def get_nv_ingest_client(config: NvidiaRAGConfig = None):
    """
    Creates and returns NV-Ingest client

    Args:
        config: NvidiaRAGConfig instance. If None, creates a new one.
    """
    if config is None:
        config = NvidiaRAGConfig()

    client = NvIngestClient(
        # Host where nv-ingest-ms-runtime is running
        message_client_hostname=config.nv_ingest.message_client_hostname,
        # REST port, defaults to 7670
        message_client_port=config.nv_ingest.message_client_port,
        message_client_kwargs={"api_version": "v2"},
    )
    return client


def get_nv_ingest_ingestor(
    nv_ingest_client_instance,
    filepaths: list[str],
    split_options=None,
    vdb_op: VDBRag = None,
    remove_extract_method: bool = False,
    extract_override: dict = None,
    config: NvidiaRAGConfig = None,
):
    """
    Prepare NV-Ingest ingestor instance based on nv-ingest configuration

    Args:
        nv_ingest_client_instance: NvIngestClient instance
        filepaths: List of file paths to ingest
        split_options: Options for splitting documents
        vdb_op: VDB operator instance (None for shallow extraction without VDB operations)
        remove_extract_method: Whether to remove extract_method from kwargs
        extract_override: Optional dict to override extraction parameters.
                         If provided, these settings override config values.
                         Useful for text-only extraction for shallow summaries.
        config: NvidiaRAGConfig instance. If None, creates a new one.

    Returns:
        ingestor: Ingestor - NV-Ingest ingestor instance with configured tasks
    """
    if config is None:
        config = NvidiaRAGConfig()

    logger.debug("Preparing NV Ingest Ingestor instance for filepaths: %s", filepaths)
    # Prepare the ingestor using nv-ingest-client
    ingestor = Ingestor(client=nv_ingest_client_instance)

    # Add files to ingestor
    ingestor = ingestor.files(filepaths)

    if config.nv_ingest.enable_pdf_split:
        ingestor = ingestor.pdf_split_config(pages_per_chunk=config.nv_ingest.pages_per_chunk)

    # Add extraction task
    # Determine table_output_format
    table_output_format = (
        "markdown" if config.nv_ingest.extract_tables else "pseudo_markdown"
    )

    # Use extract_override if provided, otherwise use config values
    if extract_override:
        extract_kwargs = extract_override.copy()
        logger.debug("Using extraction override: %s", extract_override)
    else:
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

    if remove_extract_method or config.nv_ingest.pdf_extract_method is None:
        extract_kwargs.pop("extract_method", None)
    elif "extract_method" in extract_kwargs:
        logger.info(
            f"Extract method used for ingestion: {extract_kwargs.get('extract_method', config.nv_ingest.pdf_extract_method)}"
        )
    ingestor = ingestor.extract(**extract_kwargs)

    # Add splitting task (By default only works for text documents)
    if split_options is not None:
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
            api_key=os.environ.get("NGC_API_KEY", ""),
            endpoint_url=config.nv_ingest.caption_endpoint_url,
            model_name=config.nv_ingest.caption_model_name,
        )

    # Add Embedding task (only when VDB operations are enabled)
    enable_nv_ingest_vdb_upload = (
        True  # When enabled entire ingestion would be performed using nv-ingest
    )
    if enable_nv_ingest_vdb_upload and vdb_op is not None:
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
                dimensions=config.embeddings.dimensions,
            )
        elif config.nv_ingest.image_elements_modality:
            ingestor = ingestor.embed(
                image_elements_modality=config.nv_ingest.image_elements_modality,
                endpoint_url=embedding_url,
                model_name=config.embeddings.model_name,
                dimensions=config.embeddings.dimensions,
            )
        else:
            ingestor = ingestor.embed(
                endpoint_url=embedding_url,
                model_name=config.embeddings.model_name,
                dimensions=config.embeddings.dimensions,
            )

    # Add save to disk task (only when VDB operations are enabled)
    if config.nv_ingest.save_to_disk and vdb_op is not None:
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

    # Add Vector-DB upload task (only when VDB operations are enabled)
    if enable_nv_ingest_vdb_upload and vdb_op is not None:
        ingestor = ingestor.vdb_upload(
            vdb_op=vdb_op,
            purge_results_after_upload=not config.nv_ingest.save_to_disk,
        )

    return ingestor

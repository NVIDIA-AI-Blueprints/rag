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
"""Utility functions used across different modules of the RAG.
1. filter_documents_by_confidence: Filter documents by confidence threshold.
2. utils_cache: Use this to convert unhashable args to hashable ones.
3. combine_dicts: Combines two dictionaries recursively, prioritizing values from dict_b.
4. sanitize_nim_url: Sanitize the NIM URL by adding http(s):// if missing and checking if the URL is hosted on NVIDIA's known endpoints.
5. get_metadata_configuration: Get the metadata configuration for a document.
6. prepare_custom_metadata_dataframe: Prepare custom metadata for a document and write it to a dataframe in csv format.
7. validate_filter_expr: Validate the filter expression for metadata filtering against multiple collections.
8. process_filter_expr: Process the filter expression by transforming it to the appropriate syntax for the configured vector store.
"""

import ast
import json
import logging
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from functools import wraps
from typing import Any
from uuid import uuid4

import pandas as pd
from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import Model, register_model

from nvidia_rag.utils import configuration
from nvidia_rag.utils.metadata_validation import (
    FilterExpressionParser,
    MetadataField,
    MetadataSchema,
)

logger = logging.getLogger(__name__)


def filter_documents_by_confidence(
    documents: list["Document"], confidence_threshold: float = 0.0
) -> list["Document"]:
    """
    Filter documents by confidence threshold.

    Args:
        documents: List of documents to filter
        confidence_threshold: Minimum confidence score threshold (0.0 to 1.0)

    Returns:
        list: Filtered documents that meet the confidence threshold
    """

    original_count = len(documents)

    def get_relevance_score(doc):
        """Helper function to safely extract and convert relevance score"""
        score = doc.metadata.get("relevance_score", 0.0)

        # Handle None values
        if score is None:
            return 0.0

        # Try to convert to float, return 0.0 if conversion fails
        try:
            return float(score)
        except (ValueError, TypeError):
            logger.warning(
                f"Invalid relevance_score '{score}' for document. Treating as 0.0"
            )
            return 0.0

    filtered_documents = [
        doc for doc in documents if get_relevance_score(doc) >= confidence_threshold
    ]
    filtered_count = len(filtered_documents)

    logger.info(
        f"Confidence threshold filtering: {original_count} -> {filtered_count} documents "
        f"(threshold: {confidence_threshold})"
    )

    return filtered_documents


def utils_cache(func: Callable) -> Callable:
    """Use this to convert unhashable args to hashable ones"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Convert unhashable args to hashable ones
        args_hashable = tuple(
            tuple(arg) if isinstance(arg, list | dict | set) else arg for arg in args
        )
        kwargs_hashable = {
            key: tuple(value) if isinstance(value, list | dict | set) else value
            for key, value in kwargs.items()
        }
        return func(*args_hashable, **kwargs_hashable)

    return wrapper


def combine_dicts(dict_a: dict[str, Any], dict_b: dict[str, Any]) -> dict[str, Any]:
    """Combines two dictionaries recursively, prioritizing values from dict_b.

    Args:
        dict_a: The first dictionary.
        dict_b: The second dictionary.

    Returns:
        A new dictionary with combined key-value pairs.
    """

    combined_dict = dict_a.copy()  # Start with a copy of dict_a

    for key, value_b in dict_b.items():
        if key in combined_dict:
            value_a = combined_dict[key]
            # Remove the special handling for "command"
            if isinstance(value_a, dict) and isinstance(value_b, dict):
                combined_dict[key] = combine_dicts(value_a, value_b)
            # Otherwise, replace the value from A with the value from B
            else:
                combined_dict[key] = value_b
        else:
            # Add any key not present in A
            combined_dict[key] = value_b

    return combined_dict


def sanitize_nim_url(url: str, model_name: str, model_type: str) -> str:
    """
    Sanitize the NIM URL by adding http(s):// if missing and checking if the URL is hosted on NVIDIA's known endpoints.
    """

    logger.debug(
        f"Sanitizing NIM URL: {url} for model: {model_name} of type: {model_type}"
    )

    # Construct the URL - if url does not start with http(s)://, add it
    if url and not url.startswith("http://") and not url.startswith("https://"):
        url = "http://" + url + "/v1"
        logger.info(
            f"{model_type} URL does not start with http(s)://, adding it: {url}"
        )
    # TODO: Check for v1 on url.endswith

    # Register model only if URL is hosted on NVIDIA's known endpoints
    if url.startswith("https://ai.api.nvidia.com") or url.startswith(
        "https://api.nvcf.nvidia.com"
    ):
        if model_type == "embedding":
            client = "NVIDIAEmbeddings"
        elif model_type == "chat":
            client = "ChatNVIDIA"
        elif model_type == "ranking":
            client = "NVIDIARerank"

        register_model(
            Model(
                id=model_name,
                model_type=model_type,
                client=client,
                endpoint=url,
            )
        )
        logger.info(
            f"Registering custom model {model_name} with client {client} at endpoint {url}"
        )
    return url


def get_metadata_configuration(
    collection_name: str,
    custom_metadata: list[dict[str, Any]] | None = None,
    all_file_paths: list[str] | None = None,
    metadata_schema: list[dict[str, Any]] | None = None,
    config: "configuration.NvidiaRAGConfig | None" = None,
):
    """
    Get the metadata configuration for a document.

    Args:
        collection_name: Name of the collection
        custom_metadata: User-provided metadata
        all_file_paths: List of file paths
        metadata_schema: Optional metadata schema for checking user_defined flags
        config: NvidiaRAGConfig instance. If None, creates a new one.
    """
    if config is None:
        config = configuration.NvidiaRAGConfig()

    # Create a temporary directory for custom metadata csv file
    csv_file_path = os.path.join(
        config.temp_dir,
        f"custom-metadata/{collection_name}_{str(uuid4())[:8]}/custom_metadata.csv",
    )
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

    meta_source_field, meta_fields = None, None
    meta_source_field, meta_fields = prepare_custom_metadata_dataframe(
        all_file_paths=all_file_paths,
        csv_file_path=csv_file_path,
        custom_metadata=custom_metadata or [],
        metadata_schema=metadata_schema or [],
    )

    return csv_file_path, meta_source_field, meta_fields


def prepare_custom_metadata_dataframe(
    all_file_paths: list[str],
    csv_file_path: str,
    custom_metadata: list[dict[str, Any]],
    metadata_schema: list[dict[str, Any]] | None = None,
) -> tuple[str, list[str]]:
    """
    Prepare custom metadata for a document and write it to a dataframe in csv format

    Returns:
        - meta_source_field: str - Source field name
        - all_metadata_fields: List[str] - All metadata fields
    """
    if metadata_schema is None:
        metadata_schema = []

    # Handle case where no file paths are provided (e.g., during collection deletion)
    if all_file_paths is None:
        all_file_paths = []

    # Build a map of field_name -> user_defined flag from schema
    schema_flags = {
        field["name"]: field.get("user_defined", True) for field in metadata_schema
    }

    meta_source_field = "source"
    custom_metadata_df_dict = {
        meta_source_field: all_file_paths,
    }
    # Prepare map for filename to metadata
    filename_to_metadata = {
        item["filename"]: item["metadata"] for item in custom_metadata
    }

    # Get all metadata fields from custom metadata
    all_metadata_fields = set()
    for metadata in filename_to_metadata.values():
        all_metadata_fields.update(metadata.keys())

    all_metadata_fields.add("filename")

    # Track which fields are actually included in the CSV
    included_fields = []

    for metadata_field in all_metadata_fields:
        # Skip auto-extracted fields (user_defined=False) entirely
        # Let nv-ingest extract these fields
        is_user_defined = schema_flags.get(metadata_field, True)
        if not is_user_defined:
            continue

        metadata_list = []
        for file_path in all_file_paths:
            filename = os.path.basename(file_path)
            metadata = filename_to_metadata.get(filename, {})

            if metadata_field == "filename":
                value = filename
            else:
                value = metadata.get(metadata_field, None)

            if (
                (isinstance(value, list) and len(value) == 0)
                or value is None
                or value == ""
            ):
                value = None

            metadata_list.append(value)

        # Only add column if it has values
        if metadata_list:
            custom_metadata_df_dict[metadata_field] = metadata_list
            included_fields.append(metadata_field)

    # Write to csv
    df = pd.DataFrame(custom_metadata_df_dict)
    df.to_csv(csv_file_path)

    return meta_source_field, included_fields


def validate_filter_expr(
    filter_expr: Any,
    collection_names: list[str],
    metadata_schemas: dict[str, Any],
    config: "configuration.NvidiaRAGConfig | None" = None,
) -> dict[str, Any]:
    """
    Validate the filter expression for metadata filtering against multiple collections.

    For Milvus: Validates string filter expressions against metadata schemas.
    For Elasticsearch: Validates list of dicts filter expressions.

    Args:
        filter_expr: Filter expression (string for Milvus, list of dicts for Elasticsearch)
        collection_names: List of collection names to validate against
        metadata_schemas: Dictionary mapping collection names to their metadata schemas
        config: NvidiaRAGConfig instance. If None, creates a new one.

    Returns:
        dict with keys:
        - status: Boolean indicating if validation passed
        - validated_collections: List of collections that support the filter
        - error_message: Error message if validation fails
        - details: Additional details about validation failures
    """
    if config is None:
        config = configuration.NvidiaRAGConfig()

    if config.vector_store.name == "elasticsearch":
        if isinstance(filter_expr, list):
            # Validate that it's a list of dicts
            for item in filter_expr:
                if not isinstance(item, dict):
                    logger.error(
                        f"Elasticsearch filter must be a list of dicts, found: {type(item)}"
                    )
                    return {
                        "status": False,
                        "validated_collections": [],
                        "error_message": "Elasticsearch filter must be a list of dictionaries",
                        "details": [f"Invalid item type: {type(item)}"],
                    }
            logger.debug(
                f"Elasticsearch filter validated successfully: {len(filter_expr)} filter conditions"
            )
            return {"status": True, "validated_collections": collection_names}
        elif isinstance(filter_expr, str):
            logger.warning(
                f"Elasticsearch expects list of dicts, but received string: {filter_expr}"
            )
            return {
                "status": False,
                "validated_collections": [],
                "error_message": "Elasticsearch expects list of dictionaries, not string",
                "details": ["Filter expression type mismatch"],
            }
        else:
            logger.error(
                f"Unexpected filter type for Elasticsearch: {type(filter_expr)}"
            )
            return {
                "status": False,
                "validated_collections": [],
                "error_message": f"Unexpected filter type for Elasticsearch: {type(filter_expr)}",
                "details": ["Filter expression type mismatch"],
            }

    elif config.vector_store.name == "milvus":
        if isinstance(filter_expr, str):
            logger.debug(
                f"Validating filter expression: '{filter_expr}' against {len(collection_names)} collections"
            )

            allow_partial_filtering = config.metadata.allow_partial_filtering

            validated_collections = []
            validation_errors = []

            def validate_single_collection(collection_name):
                try:
                    metadata_schema_data = metadata_schemas.get(collection_name)

                    if not metadata_schema_data:
                        return {
                            "collection": collection_name,
                            "valid": False,
                            "error": f"No metadata schema defined for collection {collection_name}",
                        }

                    # Convert raw schema data to MetadataSchema object
                    field_definitions = []
                    for field_data in metadata_schema_data:
                        field_definitions.append(MetadataField(**field_data))
                    metadata_schema = MetadataSchema(schema=field_definitions)

                    # Validate filter expression against this collection's schema
                    parser = FilterExpressionParser(metadata_schema, config)
                    result = parser.validate_filter_expression(filter_expr)

                    if result["status"]:
                        return {
                            "collection": collection_name,
                            "valid": True,
                            "error": None,
                        }
                    else:
                        return {
                            "collection": collection_name,
                            "valid": False,
                            "error": result.get(
                                "error_message", "Unknown validation error"
                            ),
                        }

                except Exception as e:
                    return {
                        "collection": collection_name,
                        "valid": False,
                        "error": str(e),
                    }

            # Execute validation in parallel
            with ThreadPoolExecutor() as executor:
                validation_results = list(
                    executor.map(validate_single_collection, collection_names)
                )

            # Process results
            for result in validation_results:
                if result["valid"]:
                    validated_collections.append(result["collection"])
                else:
                    validation_errors.append(
                        f"Collection '{result.get('collection', 'unknown collection')}': {result.get('error', 'Filter expression validation failed for this collection.')}"
                    )

            # Determine overall validation status based on allow_partial_filtering setting
            if allow_partial_filtering:
                # Flexible mode: succeed if at least one collection supports the filter
                if validated_collections:
                    logger.info(
                        f"Flexible mode: {len(validated_collections)}/{len(collection_names)} collections support filter"
                    )
                    return {
                        "status": True,
                        "validated_collections": validated_collections,
                        "details": validation_errors if validation_errors else None,
                    }
                else:
                    logger.error("No collections support the filter expression")
                    return {
                        "status": False,
                        "validated_collections": [],
                        "error_message": "No collections support the filter expression",
                        "details": validation_errors,
                    }
            else:
                # Strict mode: fail if any collection doesn't support the filter
                if len(validated_collections) < len(collection_names):
                    logger.error(
                        f"Strict mode: {len(collection_names) - len(validated_collections)} collections do not support filter"
                    )
                    return {
                        "status": False,
                        "validated_collections": validated_collections,
                        "error_message": f"Filter expression not supported by {len(collection_names) - len(validated_collections)} collections",
                        "details": validation_errors,
                    }
                else:
                    logger.debug(
                        f"All {len(collection_names)} collections support filter"
                    )
                    return {
                        "status": True,
                        "validated_collections": validated_collections,
                    }

        elif isinstance(filter_expr, list):
            logger.error("Milvus expects string filter, but received list")
            return {
                "status": False,
                "validated_collections": [],
                "error_message": "Milvus expects string filter expression, not list",
                "details": ["Filter expression type mismatch"],
            }
        else:
            logger.error(f"Unexpected filter type for Milvus: {type(filter_expr)}")
            return {
                "status": False,
                "validated_collections": [],
                "error_message": f"Unexpected filter type for Milvus: {type(filter_expr)}",
                "details": ["Filter expression type mismatch"],
            }

    else:
        logger.error(f"Unsupported vector store: {config.vector_store.name}")
        return {
            "status": False,
            "validated_collections": [],
            "error_message": f"Unsupported vector store: {config.vector_store.name}",
            "details": ["Vector store not supported"],
        }


def process_filter_expr(
    filter_expr: str | list[dict[str, Any]],
    collection_name: str = "",
    metadata_schema_data: list[dict] | None = None,
    is_generated_filter: bool = False,
    config: "configuration.NvidiaRAGConfig | None" = None,
) -> str | list[dict[str, Any]]:
    """
    Process the filter expression by transforming it to the appropriate syntax for the configured vector store.
    For Milvus: Uses FilterExpressionParser to transform user-friendly syntax to Milvus syntax.
    For Elasticsearch: Validates and returns the list of dicts format.
    Returns the processed expression or the original if processing fails.

    Args:
        filter_expr: The filter expression to process (string for Milvus, list of dicts for Elasticsearch)
        collection_name: The collection name (for logging purposes)
        metadata_schema_data: Pre-fetched metadata schema data (required for Milvus processing)
        is_generated_filter: Whether this filter was generated by LLM (affects error handling)
        config: NvidiaRAGConfig instance. If None, creates a new one.
    """
    if config is None:
        config = configuration.NvidiaRAGConfig()

    if not filter_expr or (isinstance(filter_expr, str) and filter_expr.strip() == ""):
        logger.debug("Filter expression is empty, returning empty string/list")
        return "" if config.vector_store.name == "milvus" else []

    logger.debug(
        f"Processing filter expression: '{filter_expr}' for collection '{collection_name}' with vector store '{config.vector_store.name}'"
    )

    if config.vector_store.name == "elasticsearch":
        if isinstance(filter_expr, list):
            # Validate that it's a list of dicts
            for item in filter_expr:
                if not isinstance(item, dict):
                    logger.error(
                        f"Elasticsearch filter must be a list of dicts, found: {type(item)}"
                    )
                    return []
            logger.debug(
                f"Elasticsearch filter validated successfully: {len(filter_expr)} filter conditions"
            )
            return filter_expr
        elif isinstance(filter_expr, str):
            logger.warning(
                f"Elasticsearch expects list of dicts, but received string: {filter_expr}"
            )
            return []
        else:
            logger.error(
                f"Unexpected filter type for Elasticsearch: {type(filter_expr)}"
            )
            return []

    elif config.vector_store.name == "milvus":
        if not isinstance(filter_expr, str):
            logger.error(
                f"Milvus expects string filter, but received: {type(filter_expr)}"
            )
            return ""

        # Require metadata schema for filter expression processing
        if not metadata_schema_data:
            logger.error(f"No metadata schema defined for collection {collection_name}")
            return filter_expr  # Return original if no schema exists

        # Convert raw schema data to MetadataSchema object
        try:
            field_definitions = []
            for field_data in metadata_schema_data:
                field_definitions.append(MetadataField(**field_data))
            metadata_schema = MetadataSchema(schema=field_definitions)
            logger.debug(
                f"Successfully created MetadataSchema object with {len(field_definitions)} field definitions"
            )
        except Exception as e:
            logger.error(
                f"Failed to convert metadata schema data to MetadataSchema object: {e}"
            )
            return filter_expr  # Return original if conversion fails

        parser = FilterExpressionParser(metadata_schema, config)
        result = parser.process_filter_expression(filter_expr)

        if result["status"]:
            processed_expr = result.get("processed_expression", filter_expr)
            logger.debug(
                f"Filter expression processed successfully: '{filter_expr}' -> '{processed_expr}'"
            )
            return processed_expr
        else:
            error_message = result.get("error_message", "Unknown error")
            if is_generated_filter:
                logger.warning(
                    f"Generated filter expression processing failed: {error_message}"
                )
                return ""
            else:
                raise ValueError(error_message)

    else:
        logger.error(f"Unsupported vector store: {config.vector_store.name}")
        return filter_expr if isinstance(filter_expr, str) else []


def perform_document_info_aggregation(
    existing_info_value: dict[str, Any],
    new_info_value: dict[str, Any],
) -> dict[str, Any]:
    """
    Perform document info aggregation.
    If the value is a dictionary, perform aggregation recursively.
    """
    result = {}
    all_keys = set(existing_info_value) | set(new_info_value)
    for key in all_keys:
        existing_val = existing_info_value.get(key)
        new_val = new_info_value.get(key)

        # Handle dictionaries recursively
        if isinstance(existing_val, dict) or isinstance(new_val, dict):
            result[key] = perform_document_info_aggregation(
                existing_info_value.get(key, {}), new_info_value.get(key, {})
            )
        # Handle numeric values (aggregation)
        elif isinstance(existing_val, int | float) and isinstance(new_val, int | float):
            result[key] = existing_val + new_val
        # Handle other types (prefer new value, fallback to existing)
        else:
            result[key] = (
                new_val
                if new_val is not None
                else (existing_val if existing_val is not None else 0)
            )
    return result


def get_current_timestamp() -> str:
    """Get current timestamp in ISO 8601 format."""
    return datetime.now(UTC).isoformat()


def derive_boolean_flags(doc_type_counts: dict) -> dict:
    """Derive boolean flags from document type counts."""
    return {
        "has_tables": doc_type_counts.get("table", 0) > 0,
        "has_charts": doc_type_counts.get("chart", 0) > 0,
        "has_images": doc_type_counts.get("image", 0) > 0,
    }


def create_catalog_metadata(
    description: str = "",
    tags: list[str] | None = None,
    owner: str = "",
    created_by: str = "",
    business_domain: str = "",
    status: str = "Active",
) -> dict[str, Any]:
    """Create catalog metadata dictionary for collection.

    Args:
        description: Human-readable description
        tags: List of tags for categorization
        owner: Owner team or person
        created_by: Username/email of creator
        business_domain: Business domain
        status: Collection status

    Returns:
        Dictionary containing catalog metadata with timestamps
    """
    current_time = get_current_timestamp()
    return {
        "description": description,
        "tags": tags or [],
        "owner": owner,
        "created_by": created_by,
        "business_domain": business_domain,
        "status": status,
        "date_created": current_time,
        "last_updated": current_time,
    }


def create_document_metadata(
    filepath: str,
    doc_type_counts: dict,
    total_elements: int,
    raw_text_elements_size: int,
) -> dict[str, Any]:
    """Create document metadata dictionary.

    Args:
        filepath: Full path to the document file
        doc_type_counts: Dictionary of document type counts
        total_elements: Total number of elements
        raw_text_elements_size: Size of raw text elements

    Returns:
        Dictionary containing document metadata
    """
    file_extension = os.path.splitext(os.path.basename(filepath))[1][1:] or "unknown"
    return {
        "description": "",
        "tags": [],
        "document_type": file_extension,
        "file_size": os.path.getsize(filepath),
        "date_created": get_current_timestamp(),
        "doc_type_counts": doc_type_counts,
        "total_elements": total_elements,
        "raw_text_elements_size": raw_text_elements_size,
    }

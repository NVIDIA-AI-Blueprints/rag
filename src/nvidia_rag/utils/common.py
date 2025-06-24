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
1. get_env_variable: Get an environment variable with a fallback to a default value.
2. utils_cache: Use this to convert unhashable args to hashable ones.
3. get_config: Parse the application configuration.
4. combine_dicts: Combines two dictionaries recursively, prioritizing values from dict_b.
5. sanitize_nim_url: Sanitize the NIM URL by adding http(s):// if missing and checking if the URL is hosted on NVIDIA's known endpoints.
"""

import logging
import os
from functools import wraps
from typing import Callable, Any, TYPE_CHECKING, List, Dict, Tuple
import ast

import pandas as pd

logger = logging.getLogger(__name__)

from langchain_nvidia_ai_endpoints import register_model, Model

from nvidia_rag.utils import configuration
from nvidia_rag.utils.configuration_wizard import ConfigWizard

def get_env_variable(
        variable_name: str,
        default_value: Any
    ) -> Any:
    """
    Get an environment variable with a fallback to a default value.
    Also checks if the variable is set, is not empty, and is not longer than 256 characters.

    Args:
        variable_name (str): The name of the environment variable to get

    Returns:
        Any: The value of the environment variable or the default value if the variable is not set
    """
    var = os.environ.get(variable_name)

    # Check if variable is set
    if var is None:
        logger.warning(f"Environment variable {variable_name} is not set. Using default value: {default_value}")
        var = default_value

    # Check min and max length of variable
    if len(var) > 256 or len(var) == 0:
        logger.warning(f"Environment variable {variable_name} is longer than 256 characters or empty. Using default value: {default_value}")
        var = default_value

    return var

def utils_cache(func: Callable) -> Callable:
    """Use this to convert unhashable args to hashable ones"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Convert unhashable args to hashable ones
        args_hashable = tuple(tuple(arg) if isinstance(arg, (list, dict, set)) else arg for arg in args)
        kwargs_hashable = {
            key: tuple(value) if isinstance(value, (list, dict, set)) else value
            for key, value in kwargs.items()
        }
        return func(*args_hashable, **kwargs_hashable)

    return wrapper

# @lru_cache
def get_config() -> "ConfigWizard":
    """Parse the application configuration."""
    config_file = os.environ.get("APP_CONFIG_FILE", "/dev/null")
    config = configuration.AppConfig.from_file(config_file)
    if config:
        return config
    raise RuntimeError("Unable to find configuration.")

def combine_dicts(dict_a, dict_b):
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

def sanitize_nim_url(url:str, model_name:str, model_type:str) -> str:
    """
    Sanitize the NIM URL by adding http(s):// if missing and checking if the URL is hosted on NVIDIA's known endpoints.
    """

    logger.debug(f"Sanitizing NIM URL: {url} for model: {model_name} of type: {model_type}")

    # Construct the URL - if url does not start with http(s)://, add it
    if url and not url.startswith("http://") and not url.startswith("https://"):
        url = "http://" + url + "/v1"
        logger.info(f"{model_type} URL does not start with http(s)://, adding it: {url}")

    # Register model only if URL is hosted on NVIDIA's known endpoints
    if url.startswith("https://integrate.api.nvidia.com") or \
       url.startswith("https://ai.api.nvidia.com") or \
       url.startswith("https://api.nvcf.nvidia.com"):

        if model_type == "embedding":
            client = "NVIDIAEmbeddings"
        elif model_type == "chat":
            client = "ChatNVIDIA"
        elif model_type == "ranking":
            client = "NVIDIARerank"

        register_model(Model(
            id=model_name,
            model_type=model_type,
            client=client,
            endpoint=url,
        ))
        logger.info(f"Registering custom model {model_name} with client {client} at endpoint {url}")
    return url

def prepare_custom_metadata_dataframe(
        all_file_paths: List[str],
        csv_file_path: str,
        custom_metadata: List[Dict[str, Any]]
    ) -> Tuple[str, List[str]]:
    """
    Prepare custom metadata for a document and write it to a dataframe in csv format

    Returns:
        - meta_source_field: str - Source field name
        - all_metadata_fields: List[str] - All metadata fields
    """
    meta_source_field = "source"
    custom_metadata_df_dict = {
        meta_source_field: all_file_paths,
    }
    # Prepare map for filename to metadata
    filename_to_metadata = {item["filename"]: item["metadata"] for item in custom_metadata}

    # Get all metadata fields from custom metadata
    all_metadata_fields = set()
    for metadata in filename_to_metadata.values():
        all_metadata_fields.update(metadata.keys())

    for metadata_field in all_metadata_fields:
        metadata_list = list()
        for file_path in all_file_paths:
            filename = os.path.basename(file_path)
            metadata = filename_to_metadata.get(filename, {})
            metadata_list.append(metadata.get(metadata_field, ""))
        custom_metadata_df_dict[metadata_field] = metadata_list

    # Write to csv
    df = pd.DataFrame(custom_metadata_df_dict)
    df.to_csv(csv_file_path)

    return meta_source_field, list(all_metadata_fields)

def validate_filter_expr(filter_expr: str) -> bool:
    """
    Validate the filter expression.
    """
    try:
        ast.parse(filter_expr, mode="eval")
    except Exception as e:
        if filter_expr == "": # Empty filter expression is valid
            return True
        logger.error("Error parsing filter expression: %s", e)
        return False
    return True

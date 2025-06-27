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

from typing import Any, Dict, List, Optional
import bleach

def sanitize_boolean(value: Any, field_name: str) -> bool:
    """Sanitize and convert a value to boolean.
    
    Args:
        value: The value to sanitize and convert
        field_name: Name of the field being validated
        
    Returns:
        bool: The sanitized boolean value
        
    Raises:
        ValueError: If the value cannot be converted to a boolean
    """
    value = bleach.clean(str(value), strip=True)
    try:
        return {"True": True, "False": False}[value]
    except KeyError as e:
        raise ValueError(f"{field_name} must be a boolean value (True/False)") from e

def sanitize_float(value: Any, field_name: str) -> float:
    """Sanitize and convert a value to float.
    
    Args:
        value: The value to sanitize and convert
        field_name: Name of the field being validated
        
    Returns:
        float: The sanitized float value
        
    Raises:
        ValueError: If the value cannot be converted to a float
    """
    try:
        return float(bleach.clean(str(value), strip=True))
    except ValueError as e:
        raise ValueError(f"{field_name} must be a valid number") from e

def normalize_model_info(value: Any, field_name: str) -> str:
    """Normalize model information by stripping whitespace and quotes.
    
    Args:
        value: The value to normalize
        field_name: Name of the field being validated
        
    Returns:
        str: The normalized string value
    """
    if isinstance(value, str):
        return value.strip(" ").strip('"')
    raise ValueError(f"{field_name} must be a string")

def validate_reranker_top_k(reranker_top_k: int, vdb_top_k: Optional[int], field_name: str) -> int:
    """Validate that reranker_top_k is less than or equal to vdb_top_k.
    
    Args:
        reranker_top_k: The reranker top k value to validate
        vdb_top_k: The vector database top k value to compare against
        field_name: Name of the field being validated
        
    Returns:
        int: The validated reranker_top_k value
        
    Raises:
        ValueError: If reranker_top_k is greater than vdb_top_k
    """
    if vdb_top_k is not None and reranker_top_k > vdb_top_k:
        raise ValueError(f"{field_name} must be less than or equal to vdb_top_k ({vdb_top_k})")
    return reranker_top_k

def validate_use_knowledge_base(value: Any) -> bool:
    """Direct validator for use_knowledge_base field."""
    return sanitize_boolean(value, "use_knowledge_base")

def validate_temperature(value: Any) -> float:
    """Direct validator for temperature field."""
    return sanitize_float(value, "temperature")

def validate_top_p(value: Any) -> float:
    """Direct validator for top_p field."""
    return sanitize_float(value, "top_p")

def validate_model_info(value: Any, field_name: str) -> str:
    """Direct validator for model information fields."""
    return normalize_model_info(value, field_name)

def validate_reranker_k(reranker_top_k: int, vdb_top_k: Optional[int]) -> int:
    """Direct validator for reranker_top_k field."""
    return validate_reranker_top_k(reranker_top_k, vdb_top_k, "reranker_top_k")
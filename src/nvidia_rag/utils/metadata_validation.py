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
"""Metadata validation utilities and constants."""

import hashlib
import json
import logging
import re
from collections.abc import Callable
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Literal,
    Protocol,
)

from dateutil import parser
from lark import Lark, Token, Transformer, Tree, UnexpectedInput, Visitor
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

logger = logging.getLogger(__name__)

# Global cache for expensive operations
_GRAMMAR_PARSER = None
_MODEL_CLASS_CACHE = {}


def get_grammar_parser():
    """Get the cached Lark grammar parser to avoid repeated file reading and parsing."""
    global _GRAMMAR_PARSER
    if _GRAMMAR_PARSER is None:
        grammar_file = Path(__file__).parent / "filter_grammar.lark"
        with open(grammar_file) as f:
            grammar_content = f.read()
        _GRAMMAR_PARSER = Lark(grammar_content, parser="lalr", lexer="basic")
    return _GRAMMAR_PARSER


class MetadataConfigProtocol(Protocol):
    """Protocol defining the interface for metadata configuration objects."""

    max_array_length: int
    max_string_length: int


# ============================================================================
# Custom Exceptions
# ============================================================================


class MetadataValidationError(ValueError):
    """Base exception for metadata validation errors."""

    pass


class FilterSyntaxError(MetadataValidationError):
    """Raised when filter expression syntax is invalid."""

    pass


class FilterSemanticError(MetadataValidationError):
    """Raised when filter expression semantics are invalid."""

    pass


class MetadataConfigError(MetadataValidationError):
    """Raised when metadata configuration is invalid."""

    pass


# ============================================================================
# Enums and Constants
# ============================================================================


class MetadataType(Enum):
    """Supported metadata field types."""

    STRING = "string"
    DATETIME = "datetime"
    NUMBER = "number"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    COMPARISON = "comparison"
    LOGICAL = "logical"


# Configuration constants - these should match MetadataConfig defaults
BOOLEAN_VALUES = {
    "true": True,
    "false": False,
    "1": True,
    "0": False,
    "on": True,
    "off": False,
    "True": True,
    "False": False,
    "TRUE": True,
    "FALSE": False,
    "ON": True,
    "OFF": False,
}

# Phase 2: Use MetadataType enum members for dictionary keys to avoid typos and improve autocompletion
TYPE_OPERATOR_MAPPING: dict[MetadataType, list[str]] = {
    MetadataType.STRING: [
        "==",
        "=",
        "!=",
        "like",
        "LIKE",
        "in",
        "IN",
        "not in",
        "NOT IN",
    ],
    MetadataType.DATETIME: [
        "==",
        "=",
        "!=",
        ">",
        ">=",
        "<",
        "<=",
        "between",
        "BETWEEN",
        "before",
        "BEFORE",
        "after",
        "AFTER",
    ],
    MetadataType.NUMBER: [
        "==",
        "=",
        "!=",
        ">",
        ">=",
        "<",
        "<=",
        "between",
        "BETWEEN",
        "in",
        "IN",
        "not in",
        "NOT IN",
    ],
    MetadataType.INTEGER: [
        "==",
        "=",
        "!=",
        ">",
        ">=",
        "<",
        "<=",
        "between",
        "BETWEEN",
        "in",
        "IN",
        "not in",
        "NOT IN",
    ],
    MetadataType.FLOAT: [
        "==",
        "=",
        "!=",
        ">",
        ">=",
        "<",
        "<=",
        "between",
        "BETWEEN",
        "in",
        "IN",
        "not in",
        "NOT IN",
    ],
    MetadataType.BOOLEAN: ["==", "=", "!="],
    MetadataType.ARRAY: [
        "==",
        "=",
        "!=",
        "includes",
        "INCLUDES",
        "does not include",
        "DOES NOT INCLUDE",
        "in",
        "IN",
        "not in",
        "NOT IN",
        "array_contains",
        "array_contains_all",
        "array_contains_any",
        "array_length",
    ],
}

# Mapping from array types to Python list types
ARRAY_TYPE_TO_PYTHON_TYPE: dict[MetadataType, type] = {
    MetadataType.STRING: list[str],
    MetadataType.NUMBER: list[int | float],
    MetadataType.INTEGER: list[int],
    MetadataType.FLOAT: list[float],
    MetadataType.BOOLEAN: list[bool],
}

# Derive valid array types from the mapping (single source of truth)
VALID_ARRAY_TYPES: set[MetadataType] = set(ARRAY_TYPE_TO_PYTHON_TYPE.keys())
VALID_ARRAY_TYPES_AS_STRINGS: set[str] = {
    metadata_type.value for metadata_type in VALID_ARRAY_TYPES
}

# Regex patterns for field reference parsing
FIELD_REF_PATTERN = re.compile(r'content_metadata\["([^\"]+)"\]')

# ============================================================================
# Type Mapping Constants (Phase 2)
# ============================================================================

# Mapping from MetadataType enum values to Python types
METADATA_TYPE_TO_PYTHON_TYPE: dict[MetadataType, type] = {
    MetadataType.STRING: str,
    MetadataType.DATETIME: str,
    MetadataType.NUMBER: int | float,
    MetadataType.INTEGER: int,
    MetadataType.FLOAT: float,
    MetadataType.BOOLEAN: bool,
    MetadataType.ARRAY: list[Any],
}


# Type validation helpers
def is_string_type(field_type: str) -> bool:
    """Check if field type is a string type."""
    return field_type == MetadataType.STRING.value


def is_datetime_type(field_type: str) -> bool:
    """Check if field type is a datetime type."""
    return field_type == MetadataType.DATETIME.value


def is_numeric_type(field_type: str) -> bool:
    """Check if field type is a numeric type."""
    return field_type in {
        MetadataType.NUMBER.value,
        MetadataType.INTEGER.value,
        MetadataType.FLOAT.value,
    }


def is_integer_type(field_type: str) -> bool:
    """Check if field type is an integer type."""
    return field_type == MetadataType.INTEGER.value


def is_array_type(field_type: str) -> bool:
    """Check if field type is an array type."""
    return field_type == MetadataType.ARRAY.value


def is_boolean_type(field_type: str) -> bool:
    """Check if field type is a boolean type."""
    return field_type == MetadataType.BOOLEAN.value


def get_python_type_for_metadata_type(
    field_type: str, array_type: str | None = None
) -> type:
    """Get Python type for a metadata field type."""
    if field_type == MetadataType.ARRAY.value and array_type:
        # Convert string array_type to MetadataType enum for lookup
        try:
            array_metadata_type = MetadataType(array_type)
            return ARRAY_TYPE_TO_PYTHON_TYPE.get(array_metadata_type, list[Any])
        except ValueError:
            return list[Any]

    # Convert string field_type to MetadataType enum for lookup
    try:
        metadata_type = MetadataType(field_type)
        return METADATA_TYPE_TO_PYTHON_TYPE.get(metadata_type, Any)
    except ValueError:
        return Any


# ============================================================================
# Utility Functions
# ============================================================================


def get_valid_array_types() -> set[str]:
    """Get the set of valid array types as strings."""
    return VALID_ARRAY_TYPES_AS_STRINGS


def validate_metadata_config(config: Any) -> MetadataConfigProtocol:
    """
    Validate and return a proper metadata configuration object.

    Args:
        config: Configuration object that should have metadata attributes or be a metadata config

    Returns:
        MetadataConfigProtocol: Validated metadata configuration

    Raises:
        MetadataConfigError: If the configuration is invalid
    """
    # Check if config is already a metadata config (has required attributes)
    if hasattr(config, "max_array_length") and hasattr(config, "max_string_length"):
        return config

    # Check if config has a metadata attribute that is a metadata config
    elif (
        hasattr(config, "metadata")
        and hasattr(config.metadata, "max_array_length")
        and hasattr(config.metadata, "max_string_length")
    ):
        return config.metadata

    else:
        raise MetadataConfigError(
            "Configuration must have metadata attributes (max_array_length, max_string_length) "
            "or have a metadata attribute with these properties"
        )


def _check_null_contradictions_common(nodes: list[Any], clause_type: str) -> None:
    """
    Check for contradictory IS NULL and IS NOT NULL clauses in logical expressions.

    This function inspects nodes and raises an error if a field is checked for
    both IS NULL and IS NOT NULL in the same clause.

    Args:
        nodes: List of nodes to inspect
        clause_type: Type of clause ('AND' or 'OR') for error message

    Raises:
        FilterSemanticError: If contradictory NULL checks are found
    """
    null_checks = {}

    for child in nodes:
        if hasattr(child, "data") and child.data == "is_null_comparison":
            field_val = str(child.children[0])
            op_val = str(child.children[1]).lower()
            if field_val not in null_checks:
                null_checks[field_val] = set()
            null_checks[field_val].add(op_val)

    for field, ops in null_checks.items():
        if "is null" in ops and "is not null" in ops:
            raise FilterSemanticError(
                f"Contradictory NULL logic: field '{field}' cannot be both IS NULL and IS NOT NULL in the same {clause_type} clause."
            )


class MetadataField(BaseModel):
    """
    Definition of a metadata field with comprehensive validation.

    This class represents a single field in a metadata schema, defining its name,
    type, requirements, and constraints. It provides validation for field names,
    array types, and length constraints to ensure data integrity.

    Attributes:
        name: The field name (required, non-empty string)
        type: The field data type (string, datetime, number, integer, float, boolean, array)
        required: Whether the field is mandatory (default: False)
        array_type: Type of array elements (required only for array fields)
        max_length: Maximum length constraint (for string/array fields only)
        description: Optional description of the field for documentation
    """

    name: str = Field(..., description="Field name")
    type: Literal[
        "string", "datetime", "number", "integer", "float", "boolean", "array"
    ] = Field(..., description="Field type")
    required: bool = Field(default=False, description="Whether the field is required")
    array_type: Literal["string", "number", "integer", "float", "boolean"] | None = (
        Field(
            default=None,
            description="Type of array elements (required for array fields)",
        )
    )
    max_length: int | None = Field(
        default=None, description="Maximum length for string/array fields"
    )
    description: str | None = Field(
        default=None, description="Optional description of the field for documentation"
    )

    @model_validator(mode="after")
    def validate_field(self) -> "MetadataField":
        """Comprehensive validation for all field properties."""
        self._validate_field_name()
        self._validate_array_type()
        self._validate_max_length()
        return self

    def _validate_field_name(self) -> None:
        """Validate that the field name is not empty and normalize it."""
        if not self.name or not self.name.strip():
            raise ValueError(
                "Field name cannot be empty. Please provide a valid field name."
            )
        self.name = self.name.strip()

    def _validate_array_type(self) -> None:
        """Validate array_type logic for array fields."""
        is_array_field = self.type == MetadataType.ARRAY.value
        has_array_type = self.array_type is not None

        if is_array_field:
            if not has_array_type:
                raise ValueError("array_type is required for array fields.")

            if self.array_type not in get_valid_array_types():
                valid_types = list(get_valid_array_types())
                raise ValueError(
                    f"Invalid array_type '{self.array_type}'. Valid types: {', '.join(valid_types)}"
                )
        elif has_array_type:
            raise ValueError("array_type should only be provided when type is array")

    def _validate_max_length(self) -> None:
        """Validate max_length constraints for applicable field types."""
        if self.max_length is not None:
            valid_max_length_types = {
                MetadataType.STRING.value,
                MetadataType.ARRAY.value,
            }
            if self.type not in valid_max_length_types:
                raise ValueError(
                    "max_length is only allowed for string or array fields"
                )
            if self.max_length <= 0:
                raise ValueError("max_length must be a positive integer")


class MetadataSchema(BaseModel):
    """
    Complete metadata schema definition for document collections.

    This class represents a complete metadata schema that defines the structure
    and validation rules for document metadata within a collection. It contains
    a list of field definitions and provides utilities for field lookup and
    validation.

    The schema ensures data consistency by defining field types, requirements,
    and constraints. It supports various data types including primitive types
    (string, number, boolean, datetime) and complex types (arrays with typed
    elements). Each field can be marked as required and can have length
    constraints where applicable.

    Attributes:
        schema: List of MetadataField definitions that compose the schema

    Properties:
        field_dict: Dictionary mapping field names to MetadataField objects for easy lookup
        required_fields: List of field names that are marked as required

    Example:
        schema = MetadataSchema(schema=[
            MetadataField(name="title", type="string", required=True, max_length=100),
            MetadataField(name="tags", type="array", array_type="string", max_length=50),
            MetadataField(name="created_date", type="datetime", required=True)
        ])
    """

    schema: list[MetadataField] = Field(..., description="List of field definitions")

    @field_validator("schema")
    @classmethod
    def validate_unique_field_names(cls, v: list[MetadataField]) -> list[MetadataField]:
        """Validate that field names are unique."""
        seen_names = set()
        duplicates = set()

        for field in v:
            if field.name in seen_names:
                duplicates.add(field.name)
            seen_names.add(field.name)

        if duplicates:
            raise ValueError(
                f"Duplicate field names found: {sorted(duplicates)}. Each field name must be unique in the schema."
            )
        return v

    @property
    def field_dict(self) -> dict[str, MetadataField]:
        """Get field definitions as a dictionary for easy lookup."""
        return {field.name: field for field in self.schema}

    @property
    def required_fields(self) -> list[str]:
        """Get list of required field names."""
        return [field.name for field in self.schema if field.required]


def get_cached_model_class(schema: MetadataSchema, config) -> type:
    """Get cached model class to avoid repeated dynamic class creation."""
    global _MODEL_CLASS_CACHE

    # Create a hash of the schema for cache key
    schema_dict = schema.model_dump()
    schema_str = json.dumps(schema_dict, sort_keys=True)
    cache_key = hashlib.md5(schema_str.encode()).hexdigest()

    if cache_key not in _MODEL_CLASS_CACHE:
        _MODEL_CLASS_CACHE[cache_key] = create_metadata_model_class(schema, config)
    return _MODEL_CLASS_CACHE[cache_key]


def create_metadata_model_class(schema: MetadataSchema, config) -> type:
    """
    Factory function to create a dynamic Pydantic model class for metadata validation.

    This function dynamically creates a Pydantic model based on the provided schema,
    with proper field types, validators, and model-level validation for required fields.

    Args:
        schema: The metadata schema defining the field structure
        config: Configuration object with validation settings

    Returns:
        A dynamically created Pydantic model class for metadata validation
    """
    metadata_config = validate_metadata_config(config)

    # Build field definitions and validators
    fields = {}
    field_validators = {}

    for field_def in schema.schema:
        field_type = get_python_type_for_metadata_type(
            field_def.type, field_def.array_type
        )
        field_kwargs = {
            "default": ... if field_def.required else None,
            "description": f"{field_def.type} field",
        }

        # Set max_length constraints
        if (
            is_string_type(field_def.type) or is_array_type(field_def.type)
        ) and field_def.max_length:
            field_kwargs["max_length"] = field_def.max_length
        elif is_string_type(field_def.type):
            field_kwargs["max_length"] = metadata_config.max_string_length
        elif is_array_type(field_def.type):
            field_kwargs["max_length"] = metadata_config.max_array_length

        fields[field_def.name] = (field_type | None, Field(**field_kwargs))

        # Add type-specific field validators
        if is_datetime_type(field_def.type):
            field_validators[f"validate_{field_def.name}"] = _create_datetime_validator(
                field_def.name
            )
        elif is_boolean_type(field_def.type):
            field_validators[f"validate_{field_def.name}"] = _create_boolean_validator(
                field_def.name
            )
        elif is_string_type(field_def.type):
            field_validators[f"validate_{field_def.name}"] = (
                _create_required_string_validator(field_def.name, field_def.required)
            )
        elif is_array_type(field_def.type):
            field_validators[f"validate_{field_def.name}"] = (
                _create_required_array_validator(
                    field_def.name, field_def.array_type, field_def.required
                )
            )
        elif is_numeric_type(field_def.type) and field_def.required:
            field_validators[f"validate_{field_def.name}"] = (
                _create_required_numeric_validator(field_def.name)
            )

    # Build the complete model dictionary
    model_dict = {
        "__annotations__": {k: v[0] for k, v in fields.items()},
        **{k: v[1] for k, v in fields.items()},
        **field_validators,
        "model_config": ConfigDict(
            extra="forbid", validate_assignment=True, arbitrary_types_allowed=True
        ),
    }

    # Create and return the model class
    return type("DynamicMetadataModel", (BaseModel,), model_dict)


# ============================================================================
# Datetime Utilities
# ============================================================================


class DatetimeUtility:
    """Unified utility for datetime parsing and normalization."""

    @staticmethod
    def normalize_datetime_to_utc_z(dt: datetime) -> str:
        """Normalize a datetime object to ISO8601 UTC format with 'Z' suffix."""
        dt = dt.astimezone(UTC).replace(microsecond=0)
        return dt.isoformat().replace("+00:00", "Z")

    @staticmethod
    def _strip_quotes(datetime_str: str) -> str:
        """Strip quotes from datetime string if present."""
        if datetime_str.startswith(('"', "'")) and datetime_str[0] == datetime_str[-1]:
            return datetime_str[1:-1]
        return datetime_str

    @staticmethod
    def _has_explicit_time(datetime_str: str) -> bool:
        """Check if a datetime string has explicit time components."""
        has_explicit_time = (
            "T" in datetime_str
            or ":" in datetime_str
            or "AM" in datetime_str.upper()
            or "PM" in datetime_str.upper()
        )

        if has_explicit_time:
            return True

        try:
            parsed_dt = parser.parse(datetime_str)
            return parsed_dt.hour != 0 or parsed_dt.minute != 0 or parsed_dt.second != 0
        except Exception as e:
            logger.debug(f"Failed to parse datetime '{datetime_str}': {e}")
            return False

    @classmethod
    def parse_datetime(
        cls, datetime_str: str, context: str = "general", operator: str = None
    ) -> str:
        """Parse and normalize datetime strings with context-aware handling."""
        datetime_str = cls._strip_quotes(datetime_str)

        try:
            parsed_dt = parser.parse(datetime_str)
            has_explicit_time = cls._has_explicit_time(datetime_str)

            if context == "filter" and not has_explicit_time and operator:
                if operator == ">=":
                    parsed_dt = parsed_dt.replace(
                        hour=0, minute=0, second=0, microsecond=0
                    )
                elif operator == ">":
                    parsed_dt = parsed_dt.replace(
                        hour=23, minute=59, second=59, microsecond=0
                    )
                elif operator == "<=":
                    parsed_dt = parsed_dt.replace(
                        hour=23, minute=59, second=59, microsecond=0
                    )
                elif operator == "<":
                    parsed_dt = parsed_dt.replace(
                        hour=0, minute=0, second=0, microsecond=0
                    )

            if not has_explicit_time:
                if parsed_dt.tzinfo is None:
                    parsed_dt = parsed_dt.replace(tzinfo=UTC)
                else:
                    parsed_dt = parsed_dt.astimezone(UTC)
            else:
                parsed_dt = parsed_dt.astimezone(UTC)

            normalized_dt = parsed_dt.replace(microsecond=0)
            return normalized_dt.isoformat().replace("+00:00", "Z")

        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Invalid datetime format: '{datetime_str}'. Use formats like: '2024-01-15', '2024-01-15T10:30:00', 'Jan 15, 2024'"
            ) from e

    @classmethod
    def convert_date_equality_to_between(cls, datetime_str: str) -> tuple[str, str]:
        """Convert a date-only equality filter to a BETWEEN expression covering the whole day."""
        datetime_str = cls._strip_quotes(datetime_str)

        try:
            parsed_dt = parser.parse(datetime_str)

            if parsed_dt.tzinfo is None:
                parsed_dt = parsed_dt.replace(tzinfo=UTC)
            else:
                parsed_dt = parsed_dt.astimezone(UTC)

            start_of_day = parsed_dt.replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_day = parsed_dt.replace(hour=23, minute=59, second=59, microsecond=0)

            start_str = start_of_day.isoformat().replace("+00:00", "Z")
            end_str = end_of_day.isoformat().replace("+00:00", "Z")

            return start_str, end_str

        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Invalid datetime format: '{datetime_str}'. Use formats like: '2024-01-15', '2024-01-15T10:30:00', 'Jan 15, 2024'"
            ) from e


def _create_datetime_validator(field_name: str) -> Callable:
    """Create a datetime field validator that normalizes datetime strings to ISO format."""

    @field_validator(field_name, mode="before")
    @classmethod
    def validate_datetime(cls, v: Any) -> str:
        """Validate and normalize datetime field to ISO format."""
        if v is None or v == "":
            raise ValueError(
                f"Datetime field '{field_name}' cannot be null or empty. Please provide a valid datetime value."
            )

        if isinstance(v, datetime):
            return DatetimeUtility.normalize_datetime_to_utc_z(v)

        if isinstance(v, str):
            try:
                return DatetimeUtility.parse_datetime(v, context="general")
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Invalid datetime format for field '{field_name}': '{v}'. Use formats like: '2024-01-15', '2024-01-15T10:30:00', 'Jan 15, 2024'"
                ) from e

        raise ValueError(
            f"Invalid datetime value for field '{field_name}': {v}. Expected a datetime object or string."
        )

    return validate_datetime


def _create_boolean_validator(field_name: str) -> Callable:
    """Create a boolean validator for a specific field."""

    @field_validator(field_name, mode="before")
    @classmethod
    def validate_boolean(cls, v: Any) -> bool:
        if v is None or v == "":
            raise ValueError(
                f"Boolean field '{field_name}' cannot be null or empty. Please provide a valid boolean value."
            )
        if isinstance(v, list) and len(v) == 0:
            raise ValueError(
                f"Boolean field '{field_name}' cannot be an empty array. Please provide a valid boolean value."
            )
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            v_lower = v.lower()
            if v_lower in BOOLEAN_VALUES:
                return BOOLEAN_VALUES[v_lower]
            else:
                raise ValueError(
                    f"Invalid boolean value '{v}' for field '{field_name}'. Valid values: true, false, 1, 0, on, off"
                )
        if isinstance(v, int | float):
            if v in (0, 1):
                return bool(v)
            else:
                raise ValueError(
                    f"Invalid boolean value '{v}' for field '{field_name}'. Valid values: 0, 1"
                )
        raise ValueError(
            f"Invalid boolean value '{v}' for field '{field_name}'. Expected boolean, string, or numeric value"
        )

    return validate_boolean


def _create_required_string_validator(
    field_name: str, required: bool = True
) -> Callable:
    """Create a validator for string fields with lowercase normalization."""

    @field_validator(field_name, mode="before")
    @classmethod
    def validate_required_string(cls, v: Any) -> str | None:
        if v is None:
            if required:
                raise ValueError(
                    f"Required string field '{field_name}' cannot be null. Please provide a valid value."
                )
            return None
        if isinstance(v, str):
            if not v.strip():
                if required:
                    raise ValueError(
                        f"Required string field '{field_name}' cannot be empty. Please provide a valid value."
                    )
                return None
            return v.strip().lower()
        else:
            raise ValueError(
                f"String field '{field_name}' must be a string. Got {type(v).__name__}."
            )

    return validate_required_string


def _create_required_array_validator(
    field_name: str, array_type: str = MetadataType.STRING.value, required: bool = True
) -> Callable:
    """Create a validator for array fields with lowercase normalization for strings."""

    @field_validator(field_name, mode="before")
    @classmethod
    def validate_required_array(cls, v: Any) -> list[Any] | None:
        if v is None:
            if required:
                raise ValueError(
                    f"Required array field '{field_name}' cannot be null. Please provide a valid array."
                )
            return None
        if isinstance(v, list):
            if not v:
                if required:
                    raise ValueError(
                        f"Required array field '{field_name}' cannot be empty. Please provide a valid array."
                    )
                return []
            if array_type == MetadataType.STRING.value:
                normalized_elements = []
                for element in v:
                    if not isinstance(element, str):
                        raise ValueError(
                            f"Array field '{field_name}' expects string elements, but got {type(element).__name__} with value '{element}'"
                        )
                    normalized_elements.append(element.lower())
                return normalized_elements
            else:
                return v
        else:
            raise ValueError(
                f"Array field '{field_name}' must be a list. Got {type(v).__name__}."
            )

    return validate_required_array


def _create_required_numeric_validator(field_name: str) -> Callable:
    """Create a validator for required numeric fields to ensure they're not null or empty strings."""

    @field_validator(field_name, mode="before")
    @classmethod
    def validate_required_numeric(cls, v: Any) -> float:
        if v is None:
            raise ValueError(
                f"Required numeric field '{field_name}' cannot be null. Please provide a valid numeric value."
            )
        if isinstance(v, str) and not v.strip():
            raise ValueError(
                f"Required numeric field '{field_name}' cannot be an empty string. Please provide a valid numeric value."
            )
        if isinstance(v, int | float):
            return float(v)
        elif isinstance(v, str):
            try:
                return float(v)
            except ValueError as err:
                raise ValueError(
                    f"Required numeric field '{field_name}' must be a valid number. Got '{v}'."
                ) from err
        else:
            raise ValueError(
                f"Required numeric field '{field_name}' must be a number. Got {type(v).__name__}."
            )

    return validate_required_numeric


# Type aliases for backward compatibility
MetadataValue = str | int | float | bool | list[Any]
MetadataDict = dict[str, MetadataValue]
FilterExpressionResult = dict[str, Any]


# ============================================================================
# Base Validator Classes
# ============================================================================


class BaseValidator:
    """Base class providing common validation utilities and decorators."""

    def __init__(self, metadata_schema: MetadataSchema, config):
        self.metadata_schema = metadata_schema
        self.config = validate_metadata_config(config)
        self.schema_dict = metadata_schema.field_dict

        if not metadata_schema.schema:
            logger.error(
                "No metadata schema provided - validation requires a defined schema"
            )
            raise FilterSemanticError(
                "Metadata schema is required for filter expression validation"
            )

    def _should_skip_validation(self, value_token: Any) -> bool:
        """Check if validation should be skipped for null/empty values."""
        if hasattr(value_token, "value") and value_token.value is None:
            return True
        value_str = self._extract_string_value(value_token)
        return not value_str.strip()

    def _extract_field_info(self, field_token: Any) -> tuple[str, MetadataField]:
        """Extract field name and definition from a field token."""
        if hasattr(field_token, "data") and field_token.data == "field":
            field_token = field_token.children[0]
        field_ref = str(field_token)
        match = FIELD_REF_PATTERN.match(field_ref)
        if not match:
            raise FilterSemanticError(
                f"Invalid field reference format: '{field_ref}'. Use format: content_metadata[\"field_name\"]"
            )
        field_name = match.group(1)
        if field_name not in self.schema_dict:
            available_fields = ", ".join(sorted(self.schema_dict.keys()))
            raise FilterSemanticError(
                f"Field '{field_name}' not found in metadata schema. Available fields: {available_fields}"
            )
        return field_name, self.schema_dict[field_name]

    def _extract_string_value(self, value_token: Any) -> str:
        """Extract string value from various token types."""
        # Handle array literal trees - return "[]" for empty arrays
        if hasattr(value_token, "data") and value_token.data == "array_literal":
            # Check if this is an empty array: [LBRACK, None, RBRACK] or [LBRACK, RBRACK]
            if (len(value_token.children) == 3 and value_token.children[1] is None) or (
                len(value_token.children) == 2
                and hasattr(value_token.children[0], "type")
                and value_token.children[0].type == "LBRACK"
                and hasattr(value_token.children[1], "type")
                and value_token.children[1].type == "RBRACK"
            ):
                return "[]"
            # For non-empty arrays, reconstruct the JSON string
            try:
                # Extract array elements and reconstruct as JSON
                elements = self._extract_array_elements_from_tree(
                    value_token, expected_type="string"
                )
                json_str = json.dumps(elements)
                return json_str
            except Exception:
                # Fall back to default logic
                pass
        elif (
            hasattr(value_token, "data")
            and value_token.data == "value"
            and value_token.children
        ):
            # Check if the first child is an array_literal
            first_child = value_token.children[0]
            if hasattr(first_child, "data") and first_child.data == "array_literal":
                # Check if this is an empty array
                if (
                    len(first_child.children) == 3 and first_child.children[1] is None
                ) or (
                    len(first_child.children) == 2
                    and hasattr(first_child.children[0], "type")
                    and first_child.children[0].type == "LBRACK"
                    and hasattr(first_child.children[1], "type")
                    and first_child.children[1].type == "RBRACK"
                ):
                    return "[]"

        if hasattr(value_token, "value"):
            if value_token.value is None:
                return ""
            return DatetimeUtility._strip_quotes(str(value_token.value))
        elif (
            hasattr(value_token, "children")
            and value_token.children
            and len(value_token.children) > 0
        ):
            return self._extract_string_value(value_token.children[0])
        else:
            return DatetimeUtility._strip_quotes(str(value_token))

    def _is_array_literal(self, value_token: Any) -> bool:
        """Check if a parser token represents an array literal."""
        if hasattr(value_token, "data") and value_token.data == "array_literal":
            return True
        elif (
            hasattr(value_token, "data")
            and value_token.data == "value"
            and value_token.children
        ):
            # Check if the first child is an array_literal
            first_child = value_token.children[0]
            if hasattr(first_child, "data") and first_child.data == "array_literal":
                return True
        return False


class ValueValidator(BaseValidator):
    """Handles value format and type validation."""

    def validate_integer_value(self, field_name: str, value_str: str) -> None:
        """Validate that a value is a valid integer."""

        try:
            float_val = float(value_str)
            if float_val != int(float_val):
                raise FilterSemanticError(
                    f"Invalid integer format for field '{field_name}': '{value_str}'. Value must be a whole number."
                )
        except ValueError as e:
            raise FilterSemanticError(
                f"Invalid integer format for field '{field_name}': '{value_str}'. Use whole numbers like: 42, -10, 0"
            ) from e

    def validate_string_value(
        self, field_name: str, value_str: str, value_token: Any
    ) -> None:
        """Validate that a value is a valid string."""

        if self._is_array_literal(value_token):
            raise FilterSemanticError(
                f"Cannot compare string field '{field_name}' to array value. Use string values for string field comparisons."
            )
        max_length = self.config.max_string_length
        if len(value_str) > max_length:
            raise FilterSemanticError(
                f"String value for field '{field_name}' is too long ({len(value_str)} characters). Maximum allowed: {max_length} characters"
            )

    def validate_datetime_value(self, value_str: str, field_name: str) -> None:
        """Validate that a value is a valid datetime string."""
        try:
            parser.parse(value_str)
        except (ValueError, TypeError) as e:
            raise FilterSemanticError(
                f"Invalid datetime format for field '{field_name}': '{value_str}'. Use formats like: '2024-01-15', '2024-01-15T10:30:00', 'Jan 15, 2024'"
            ) from e

    def validate_number_value(
        self, field_name: str, value_str: str, value_token: Any
    ) -> None:
        """Validate that a value is a valid number."""

        if self._is_array_literal(value_token):
            raise FilterSemanticError(
                f"Cannot compare number field '{field_name}' to array value. Use numeric values for number field comparisons."
            )
        try:
            float(value_str)
        except ValueError as e:
            raise FilterSemanticError(
                f"Invalid number format for field '{field_name}': '{value_str}'. Use numeric values like: 42, 3.14, -10"
            ) from e

    def validate_boolean_value(
        self, field_name: str, value_str: str, value_token: Any
    ) -> None:
        """Enhanced validation for boolean fields to catch string representations and NULL operations."""

        if value_str not in BOOLEAN_VALUES:
            valid_values = ", ".join(sorted(BOOLEAN_VALUES.keys()))
            raise FilterSemanticError(
                f"Cannot compare boolean field '{field_name}' to non-boolean value '{value_str}'. "
                f"Use boolean values: {valid_values}"
            )

        if hasattr(value_token, "children") and value_token.children:
            for child in value_token.children:
                if hasattr(child, "type") and child.type == "ESCAPED_STRING":
                    value_str = self._extract_string_value(child)
                    if value_str.lower() in BOOLEAN_VALUES:
                        raise FilterSemanticError(
                            f"Boolean field '{field_name}' cannot use quoted string '{value_str}'. "
                            f"Use unquoted boolean values: true, false, True, False, TRUE, FALSE, 1, 0, on, off, ON, OFF"
                        )
                    break

                if hasattr(child, "type") and child.type == "BOOLEAN":
                    value_str = str(child.value)
                    if value_str.lower() not in BOOLEAN_VALUES:
                        valid_values = ", ".join(sorted(BOOLEAN_VALUES.keys()))
                        raise FilterSemanticError(
                            f"Invalid boolean value '{value_str}' for field '{field_name}'. "
                            f"Use: {valid_values}"
                        )
                    break

    def validate_value_type_for_field(
        self, field_name: str, field_info: MetadataField, value_token: Any
    ) -> None:
        """Validate that the value type is compatible with the field type."""
        if self._should_skip_validation(value_token):
            return

        field_type = field_info.type
        value_str = self._extract_string_value(value_token)
        value_raw = getattr(value_token, "value", None)

        is_quoted_string = False
        if isinstance(value_token, Tree) and value_token.children:
            child = value_token.children[0]
            if isinstance(child, Token) and child.type == "ESCAPED_STRING":
                is_quoted_string = True

        if not value_str.strip():
            return

        if value_str.lower() == "null":
            raise FilterSemanticError(
                f"NULL operations are not supported for {field_type} fields like '{field_name}'. "
            )

        if is_string_type(field_type):
            if is_quoted_string:
                return
            if isinstance(value_raw, int | float):
                raise FilterSemanticError(
                    f"Cannot compare string field '{field_name}' to numeric value '{value_raw}'. "
                    f"Use string values for string field comparisons."
                )
            is_numeric = False
            try:
                float(value_str)
                is_numeric = True
            except ValueError:
                pass
            if is_numeric:
                raise FilterSemanticError(
                    f"Cannot compare string field '{field_name}' to numeric value '{value_str}'. "
                    f"Use string values for string field comparisons."
                )
            self.validate_string_value(field_name, value_str, value_token)

        elif is_integer_type(field_type):
            if is_quoted_string:
                raise FilterSemanticError(
                    f"Cannot compare integer field '{field_name}' to string value '{value_str}'. "
                    f"Use unquoted integer values for integer field comparisons."
                )
            self.validate_integer_value(field_name, value_str)

        elif is_numeric_type(field_type):
            if is_quoted_string:
                raise FilterSemanticError(
                    f"Cannot compare {field_type} field '{field_name}' to string value '{value_str}'. "
                    f"Use unquoted numeric values for {field_type} field comparisons."
                )
            self.validate_number_value(field_name, value_str, value_token)

        elif is_boolean_type(field_type):
            self.validate_boolean_value(field_name, value_str, value_token)

        elif is_datetime_type(field_type):
            if isinstance(value_raw, int | float):
                raise FilterSemanticError(
                    f"Cannot compare datetime field '{field_name}' to numeric value '{value_raw}'. "
                    f"Use datetime string values for datetime field comparisons."
                )

            is_numeric = False
            try:
                float(value_str)
                is_numeric = True
            except ValueError:
                pass
            if is_numeric:
                raise FilterSemanticError(
                    f"Cannot compare datetime field '{field_name}' to numeric value '{value_str}'. "
                    f"Use datetime string values for datetime field comparisons."
                )
            self.validate_datetime_value(value_str, field_name)


class OperatorValidator(BaseValidator):
    """Handles operator compatibility validation."""

    def validate_operator_for_type(
        self, operator: str, field_type: str, field_name: str
    ) -> None:
        """Validate that an operator is supported for a given field type."""
        # Convert string field_type to MetadataType enum for lookup
        try:
            metadata_type = MetadataType(field_type)
        except ValueError as e:
            supported_types = ", ".join([t.value for t in TYPE_OPERATOR_MAPPING.keys()])
            raise FilterSemanticError(
                f"Field type '{field_type}' is not supported for field '{field_name}'. Supported types: {supported_types}"
            ) from e

        if metadata_type not in TYPE_OPERATOR_MAPPING:
            supported_types = ", ".join([t.value for t in TYPE_OPERATOR_MAPPING.keys()])
            raise FilterSemanticError(
                f"Field type '{field_type}' is not supported for field '{field_name}'. Supported types: {supported_types}"
            )

        supported_ops = TYPE_OPERATOR_MAPPING[metadata_type]
        if operator not in supported_ops:
            supported_ops_str = ", ".join(supported_ops)
            raise FilterSemanticError(
                f"Operator '{operator}' is not supported for {field_type} fields like '{field_name}'. Use one of: {supported_ops_str}"
            )


class FieldValidator(BaseValidator):
    """Handles field existence and reference validation."""

    def validate_field_existence(self, field_token: Any) -> tuple[str, MetadataField]:
        """Validate field existence in schema and reject array indexing."""
        field_ref = str(field_token)
        array_index_pattern = re.compile(r'content_metadata\["[^"]+"\]\s*\[\s*\d+\s*\]')
        if array_index_pattern.match(field_ref):
            raise FilterSemanticError(
                'Array indexing (e.g., content_metadata["tags"][0]) is not supported. '
                "Use array functions like array_contains instead."
            )
        try:
            return self._extract_field_info(field_token)
        except FilterSemanticError as e:
            raise FilterSemanticError(f"Invalid filter field: {str(e)}") from e


class ArrayValidator(BaseValidator):
    """Handles array-specific validation logic."""

    def _extract_array_elements_from_tree(
        self, tree, expected_type=None, require_quoted_strings=False
    ) -> list:
        """Extract array elements from a parser tree structure."""

        def extract_primitive(el):
            """Extract primitive value from a tree element."""
            was_quoted = False

            while isinstance(el, Tree) and hasattr(el, "children") and el.children:
                el = el.children[0]

            if isinstance(el, Token):
                if el.type == "ESCAPED_STRING":
                    was_quoted = True
                    result = str(el.value)[1:-1]
                    return (result, was_quoted) if require_quoted_strings else result

                if el.type == "SIGNED_NUMBER":
                    try:
                        if expected_type == MetadataType.STRING.value:
                            result = str(el.value)
                            return (
                                (result, was_quoted)
                                if require_quoted_strings
                                else result
                            )
                        if "." in str(el.value):
                            result = float(el.value)
                            return (
                                (result, was_quoted)
                                if require_quoted_strings
                                else result
                            )
                        result = int(el.value)
                        return (
                            (result, was_quoted) if require_quoted_strings else result
                        )
                    except Exception:
                        return (
                            (el.value, was_quoted)
                            if require_quoted_strings
                            else el.value
                        )

                if el.type == "BOOLEAN":
                    val = str(el.value).lower()
                    if val in BOOLEAN_VALUES:
                        result = BOOLEAN_VALUES[val]
                        return (
                            (result, was_quoted) if require_quoted_strings else result
                        )
                    return (
                        (el.value, was_quoted) if require_quoted_strings else el.value
                    )

                result = str(el.value)
                return (result, was_quoted) if require_quoted_strings else result

            elif hasattr(el, "value"):
                result = el.value
                return (result, was_quoted) if require_quoted_strings else result

            if isinstance(el, str):
                s = el.strip("\"'")
                if expected_type == MetadataType.STRING.value:
                    return (s, was_quoted) if require_quoted_strings else s
                return (s, was_quoted) if require_quoted_strings else s

            return (el, was_quoted) if require_quoted_strings else el

        if hasattr(tree, "data") and tree.data == "array_literal":
            if (len(tree.children) == 3 and tree.children[1] is None) or (
                len(tree.children) == 2
                and isinstance(tree.children[0], Token)
                and isinstance(tree.children[1], Token)
            ):
                return []

            for child in tree.children:
                if hasattr(child, "data") and child.data == "array_elements":
                    elements = []
                    for grandchild in child.children:
                        if (
                            isinstance(grandchild, Tree)
                            and getattr(grandchild, "data", None) == "array_element"
                        ):
                            element = extract_primitive(grandchild)
                            elements.append(element)
                    return elements

        if hasattr(tree, "children") and tree.children:
            for child in tree.children:
                if hasattr(child, "data") and child.data == "array_literal":
                    return self._extract_array_elements_from_tree(
                        child,
                        expected_type=expected_type,
                        require_quoted_strings=require_quoted_strings,
                    )

        if isinstance(tree, list):
            return tree

        return []

    def _validate_array_element_type(self, element: Any, expected_type: str) -> bool:
        """Strictly validate that an array element matches the expected type."""
        if expected_type == MetadataType.STRING.value:
            return isinstance(element, str)
        if expected_type == MetadataType.INTEGER.value:
            return isinstance(element, int)
        if expected_type in (MetadataType.FLOAT.value, MetadataType.NUMBER.value):
            return isinstance(element, int | float) and not isinstance(element, bool)
        if expected_type == MetadataType.BOOLEAN.value:
            return isinstance(element, bool)
        return False

    def _validate_array_contents(self, elements: list[Any], field_name: str) -> None:
        """Validate array contents for common issues."""
        if not elements:
            return

        for el in elements:
            if el == "":
                raise FilterSemanticError(
                    f"Empty string elements are not supported in array filters for field '{field_name}', because empty strings are never ingested. Please remove empty strings from your filter."
                )

        if elements:
            first_type = type(elements[0])
            first_is_numeric = isinstance(elements[0], int | float)

            for i, el in enumerate(elements):
                current_type = type(el)
                current_is_numeric = isinstance(el, int | float)

                if first_type != current_type:
                    if first_is_numeric and current_is_numeric:
                        continue
                    else:
                        raise FilterSemanticError(
                            f"Mixed-type arrays are not supported for field '{field_name}'. All elements must be of the same type. Found {first_type.__name__} and {current_type.__name__} at index {i}."
                        )

                if isinstance(el, float) and (
                    el != el or el == float("inf") or el == float("-inf")
                ):
                    raise FilterSemanticError(
                        f"Special float values (NaN, inf) are not supported in array filters for field '{field_name}'."
                    )

    def validate_array_value(self, field_name: str, value_token: Any) -> None:
        """Validate that a value is a valid array."""
        if self._should_skip_validation(value_token):
            return

        field_info = self.schema_dict.get(field_name)
        if (
            not field_info
            or not is_array_type(field_info.type)
            or not field_info.array_type
        ):
            return

        elements = self._extract_array_elements_from_tree(
            value_token, expected_type=field_info.array_type
        )
        self._validate_array_contents(elements, field_name)

        for i, element in enumerate(elements):
            if not self._validate_array_element_type(element, field_info.array_type):
                raise FilterSemanticError(
                    f"Array element at index {i} for field '{field_name}' has invalid type. Expected {field_info.array_type}, got {type(element).__name__}"
                )

        value_str = self._extract_string_value(value_token)
        max_length = self.config.max_array_length
        if field_info.max_length is not None:
            max_length = field_info.max_length

        if value_str.strip() == "[]" or value_str.strip() == "":
            return

        array_value = None
        try:
            array_value = json.loads(value_str)
        except Exception:
            pass

        if not isinstance(array_value, list):
            raise FilterSemanticError(
                f"Array value for field '{field_name}' must be a list."
            )
        if len(array_value) > max_length:
            raise FilterSemanticError(
                f"Array value for field '{field_name}' exceeds max_length {max_length}."
            )

        self._validate_array_contents(array_value, field_name)


class FilterSemanticValidator(Visitor):
    """Semantic validator for filter expressions using visitor pattern."""

    def __init__(self, metadata_schema: MetadataSchema, config):
        self.value_validator = ValueValidator(metadata_schema, config)
        self.operator_validator = OperatorValidator(metadata_schema, config)
        self.field_validator = FieldValidator(metadata_schema, config)
        self.array_validator = ArrayValidator(metadata_schema, config)

        self.metadata_schema = metadata_schema
        self.schema_dict = metadata_schema.field_dict
        self.config = validate_metadata_config(config)

    def field(self, tree) -> Any:
        """Validate field existence in schema and reject array indexing."""
        field_token = tree.children[0]
        self.field_validator.validate_field_existence(field_token)
        return tree

    def comparison(self, tree) -> Any:
        """Validate comparison operations for field types."""
        field_token = tree.children[0]
        op_token = tree.children[1]
        value_token = tree.children[2] if len(tree.children) > 2 else None

        field_name, field_info = self.field_validator.validate_field_existence(
            field_token
        )
        field_type = field_info.type
        op_val = str(op_token)

        self.operator_validator.validate_operator_for_type(
            op_val, field_type, field_name
        )

        if value_token is not None and not is_array_type(field_type):
            self.value_validator.validate_value_type_for_field(
                field_name, field_info, value_token
            )
        elif is_array_type(field_type):
            self.array_validator.validate_array_value(field_name, value_token)

        return tree

    def like_comparison(self, tree) -> Any:
        """Validate that LIKE operations are only allowed on string fields."""
        field_name, field_info = self.field_validator.validate_field_existence(
            tree.children[0]
        )

        self.operator_validator.validate_operator_for_type(
            "like", field_info.type, field_name
        )

        if not is_string_type(field_info.type):
            raise FilterSemanticError(
                f"LIKE operator is only allowed on string fields, not '{field_info.type}'"
            )
        return tree

    def array_function(self, tree) -> Any:
        """Validate that array functions are only used on array fields."""
        func_name = str(tree.children[0])
        if func_name.lower() == "array_length":
            if len(tree.children) != 4:
                raise FilterSemanticError(
                    f"array_length function expects exactly one argument (the field name), got: {len(tree.children) - 2}"
                )
            field_token = tree.children[2]
        else:
            field_token = tree.children[2]

        field_name, field_info = self.field_validator.validate_field_existence(
            field_token
        )
        if not is_array_type(field_info.type):
            raise FilterSemanticError(
                f"Array function '{func_name}' cannot be used on field '{field_name}' (type: {field_info.type}). Array functions only work with array fields."
            )
        return tree

    def array_length_comparison(self, tree) -> Any:
        """Validate that array_length is only used on array fields."""
        field_name, field_info = self.field_validator.validate_field_existence(
            tree.children[2]
        )
        if not is_array_type(field_info.type):
            raise FilterSemanticError(
                f"array_length function cannot be used on field '{field_name}' (type: {field_info.type}). array_length only works with array fields."
            )

        value_token = tree.children[5]
        value_str = self.value_validator._extract_string_value(value_token)

        try:
            int_val = int(value_str)
            if str(int_val) != value_str:
                raise ValueError()
        except Exception as e:
            raise FilterSemanticError(
                f"array_length comparison value must be an integer, got '{value_str}'"
            ) from e

        if int_val < 0:
            raise FilterSemanticError(
                f"array_length comparison value must be non-negative, got {int_val}"
            )

        max_length = self.config.max_array_length
        if field_info.max_length is not None:
            max_length = field_info.max_length
        if int_val > max_length:
            raise FilterSemanticError(
                f"array_length comparison value {int_val} exceeds max_length {max_length} for field '{field_name}'"
            )
        return tree

    def array_membership(self, tree) -> Any:
        """Validate that array membership is only used on array fields."""
        field_name, field_info = self.field_validator.validate_field_existence(
            tree.children[2]
        )
        if not is_array_type(field_info.type):
            raise FilterSemanticError(
                f"Array membership operations (like 'in') can only be used on array fields, not '{field_info.type}' fields like '{field_name}'."
            )
        return tree

    def array_membership_negated(self, tree) -> Any:
        """Validate that negated array membership is only used on array fields."""
        field_name, field_info = self.field_validator.validate_field_existence(
            tree.children[2]
        )
        if not is_array_type(field_info.type):
            raise FilterSemanticError(
                f"Array membership operations (like 'not in') can only be used on array fields, not '{field_info.type}' fields like '{field_name}'."
            )
        return tree

    def array_comparison(self, tree) -> Any:
        field_name, field_info = self.field_validator.validate_field_existence(
            tree.children[0]
        )
        op_token = tree.children[1]
        value_token = tree.children[2]
        op_val = str(op_token).lower()

        if str(value_token).strip() == "[]":
            raise FilterSemanticError(
                f"Empty array comparisons are not supported for field '{field_name}'. "
                f"Please provide non-empty array values or use explicit field comparisons."
            )

        self.operator_validator.validate_operator_for_type(
            op_val, field_info.type, field_name
        )

        if op_val in ["in", "not in"]:
            if not (
                is_string_type(field_info.type)
                or is_array_type(field_info.type)
                or is_numeric_type(field_info.type)
                or is_datetime_type(field_info.type)
            ):
                raise FilterSemanticError(
                    f"'{op_val}' operator can only be used on string, array, or numeric fields, not '{field_info.type}' fields like '{field_name}'."
                )

            value_str = str(value_token).strip()
            if not (value_str.startswith("[") or "array_literal" in str(value_token)):
                raise FilterSemanticError(
                    f"'{op_val}' operator requires an array literal as the right operand. Example: content_metadata[\"{field_name}\"] {op_val} ['item1', 'item2']"
                )

            if is_array_type(field_info.type) and field_info.array_type:
                require_quoted = field_info.array_type == MetadataType.STRING.value
                elements = self.array_validator._extract_array_elements_from_tree(
                    value_token,
                    expected_type=field_info.array_type,
                    require_quoted_strings=require_quoted,
                )

                if not elements:
                    return tree

                if require_quoted:
                    for i, (element, was_quoted) in enumerate(elements):
                        if not was_quoted:
                            raise FilterSemanticError(
                                f"Array element at index {i} for field '{field_name}' must be a quoted string (e.g., \"value\"). Unquoted values are not allowed in string array filters."
                            )
                        if element == "":
                            raise FilterSemanticError(
                                f"Empty string elements are not supported in array filters for field '{field_name}', because empty strings are never ingested. Please remove empty strings from your filter."
                            )
                else:
                    for i, element in enumerate(elements):
                        if element == "":
                            raise FilterSemanticError(
                                f"Empty string elements are not supported in array filters for field '{field_name}', because empty strings are never ingested. Please remove empty strings from your filter."
                            )

                if not require_quoted and elements:
                    first_type = type(elements[0])
                    first_is_numeric = isinstance(elements[0], int | float)
                    for i, element in enumerate(elements):
                        current_type = type(element)
                        current_is_numeric = isinstance(element, int | float)
                        if first_type != current_type:
                            if first_is_numeric and current_is_numeric:
                                continue
                            else:
                                raise FilterSemanticError(
                                    f"Mixed-type arrays are not supported for field '{field_name}'. All elements must be of the same type. Found {first_type.__name__} and {current_type.__name__} at index {i}."
                                )
                        if isinstance(element, float) and (
                            element != element
                            or element == float("inf")
                            or element == float("-inf")
                        ):
                            raise FilterSemanticError(
                                f"Special float values (NaN, inf) are not supported in array filters for field '{field_name}'."
                            )
                    for i, element in enumerate(elements):
                        if not self.array_validator._validate_array_element_type(
                            element, field_info.array_type
                        ):
                            raise FilterSemanticError(
                                f"Array element at index {i} for field '{field_name}' has invalid type. Expected {field_info.array_type}, got {type(element).__name__}"
                            )
        return tree

    def between_comparison(self, tree) -> Any:
        """Validate that between operations are only allowed on appropriate field types."""
        field_name, field_info = self.field_validator.validate_field_existence(
            tree.children[0]
        )
        start_token = tree.children[2]
        end_token = tree.children[4]

        self.operator_validator.validate_operator_for_type(
            "between", field_info.type, field_name
        )

        if is_string_type(field_info.type):
            raise FilterSemanticError(
                f"BETWEEN operator is not supported for string fields like '{field_name}'. Use LIKE for pattern matching or IN for membership testing."
            )

        if is_boolean_type(field_info.type):
            raise FilterSemanticError(
                f"BETWEEN operator is not supported for boolean fields like '{field_name}'. Use == or != for boolean comparisons."
            )

        if is_array_type(field_info.type):
            raise FilterSemanticError(
                f"BETWEEN operator is not supported for array fields like '{field_name}'. Use array functions like array_contains or array_length for array operations."
            )

        if is_datetime_type(field_info.type):
            start_str = self.value_validator._extract_string_value(start_token)
            end_str = self.value_validator._extract_string_value(end_token)

            try:
                start_dt = parser.parse(start_str)
            except (ValueError, TypeError) as e:
                raise FilterSemanticError(
                    f"Invalid datetime format for field '{field_name}' start value: '{start_str}'. Use formats like: '2024-01-01', '2024-01-01T00:00:00'"
                ) from e

            try:
                end_dt = parser.parse(end_str)
            except (ValueError, TypeError) as e:
                raise FilterSemanticError(
                    f"Invalid datetime format for field '{field_name}' end value: '{end_str}'. Use formats like: '2024-12-31', '2024-12-31T23:59:59'"
                ) from e

            if start_dt > end_dt:
                raise FilterSemanticError(
                    f"Invalid BETWEEN range for field '{field_name}': start value '{start_str}' is after end value '{end_str}'. "
                    f"The start value must be less than or equal to the end value."
                )

        elif is_numeric_type(field_info.type):
            start_str = self.value_validator._extract_string_value(start_token)
            end_str = self.value_validator._extract_string_value(end_token)

            try:
                start_val = float(start_str)
            except ValueError as e:
                raise FilterSemanticError(
                    f"Invalid numeric format for field '{field_name}' start value: '{start_str}'. Use numeric values like: 10, 5.5, -3"
                ) from e

            try:
                end_val = float(end_str)
            except ValueError as e:
                raise FilterSemanticError(
                    f"Invalid numeric format for field '{field_name}' end value: '{end_str}'. Use numeric values like: 100, 10.5, -1"
                ) from e

            if start_val > end_val:
                raise FilterSemanticError(
                    f"Invalid BETWEEN range for field '{field_name}': start value '{start_str}' is greater than end value '{end_str}'. "
                    f"The start value must be less than or equal to the end value."
                )

        return tree

    def is_null_comparison(self, tree) -> Any:
        """Reject all IS NULL operations as they are not supported."""
        field_name, field_info = self.field_validator.validate_field_existence(
            tree.children[0]
        )
        op_token = tree.children[1]
        op_val = str(op_token).lower()

        raise FilterSemanticError(
            f"NULL operations (IS NULL, IS NOT NULL) are not supported in JSON metadata fields. "
            f"Field '{field_name}' cannot use '{op_val}'. "
            f"Please provide explicit values for all fields."
        )

    def _check_null_contradictions(self, children: list[Any], clause_type: str) -> None:
        """Check for contradictory IS NULL and IS NOT NULL clauses in logical expressions."""
        _check_null_contradictions_common(children, clause_type)

    def and_expr(self, tree) -> Any:
        """Validate AND expression and check for null contradictions."""
        self._check_null_contradictions(tree.children, "AND")
        return tree

    def or_expr(self, tree) -> Any:
        """Validate OR expression and check for null contradictions."""
        self._check_null_contradictions(tree.children, "OR")
        return tree

    def _validate_field_in_list_common(
        self, field_token, value_token, operator="in", tree=None
    ) -> Any:
        """Common validation logic for field in/not in list operations."""
        field_name, field_info = self.field_validator.validate_field_existence(
            field_token
        )

        self.operator_validator.validate_operator_for_type(
            operator, field_info.type, field_name
        )

        elements = []
        if is_array_type(field_info.type) and field_info.array_type:
            require_quoted = field_info.array_type == MetadataType.STRING.value
            elements = self.array_validator._extract_array_elements_from_tree(
                value_token,
                expected_type=field_info.array_type,
                require_quoted_strings=require_quoted,
            )

            if not elements:
                return tree

            if require_quoted:
                for i, (element, was_quoted) in enumerate(elements):
                    if not was_quoted:
                        raise FilterSemanticError(
                            f"Array element at index {i} for field '{field_name}' must be a quoted string (e.g., \"value\"). Unquoted values are not allowed in string array filters."
                        )
                    if element == "":
                        raise FilterSemanticError(
                            f"Empty string elements are not supported in array filters for field '{field_name}', because empty strings are never ingested. Please remove empty strings from your filter."
                        )
            else:
                for i, element in enumerate(elements):
                    if element == "":
                        raise FilterSemanticError(
                            f"Empty string elements are not supported in array filters for field '{field_name}', because empty strings are never ingested. Please remove empty strings from your filter."
                        )

            if not require_quoted and elements:
                first_type = type(elements[0])
                first_is_numeric = isinstance(elements[0], int | float)
                for i, element in enumerate(elements):
                    current_type = type(element)
                    current_is_numeric = isinstance(element, int | float)
                    if first_type != current_type:
                        if first_is_numeric and current_is_numeric:
                            continue
                        else:
                            raise FilterSemanticError(
                                f"Mixed-type arrays are not supported for field '{field_name}'. All elements must be of the same type. Found {first_type.__name__} and {current_type.__name__} at index {i}."
                            )
                    if isinstance(element, float) and (
                        element != element
                        or element == float("inf")
                        or element == float("-inf")
                    ):
                        raise FilterSemanticError(
                            f"Special float values (NaN, inf) are not supported in array filters for field '{field_name}'."
                        )
                for i, element in enumerate(elements):
                    if not self.array_validator._validate_array_element_type(
                        element, field_info.array_type
                    ):
                        raise FilterSemanticError(
                            f"Array element at index {i} for field '{field_name}' has invalid type. Expected {field_info.array_type}, got {type(element).__name__}"
                        )
        else:
            elements = self.array_validator._extract_array_elements_from_tree(
                value_token, expected_type=None, require_quoted_strings=False
            )

            if not elements:
                return tree

            for i, element in enumerate(elements):
                if (
                    is_string_type(field_info.type)
                    and isinstance(element, tuple)
                    and len(element) == 2
                ):
                    element_value, was_quoted = element
                    if not isinstance(element_value, str):
                        raise FilterSemanticError(
                            f"Array element at index {i} for string field '{field_name}' must be a string. Got {type(element_value).__name__} with value '{element_value}'."
                        )
                elif is_string_type(field_info.type):
                    if not isinstance(element, str):
                        raise FilterSemanticError(
                            f"Array element at index {i} for string field '{field_name}' must be a string. Got {type(element).__name__} with value '{element}'."
                        )
                elif is_numeric_type(field_info.type):
                    if not isinstance(element, int | float) or isinstance(
                        element, bool
                    ):
                        raise FilterSemanticError(
                            f"Array element at index {i} for {field_info.type} field '{field_name}' must be a number. Got {type(element).__name__} with value '{element}'."
                        )
                elif is_boolean_type(field_info.type):
                    if not isinstance(element, bool):
                        raise FilterSemanticError(
                            f"Array element at index {i} for boolean field '{field_name}' must be a boolean. Got {type(element).__name__} with value '{element}'."
                        )
                elif is_datetime_type(field_info.type):
                    if not isinstance(element, str):
                        raise FilterSemanticError(
                            f"Array element at index {i} for datetime field '{field_name}' must be a string. Got {type(element).__name__} with value '{element}'."
                        )
                    try:
                        parser.parse(element)
                    except (ValueError, TypeError) as e:
                        raise FilterSemanticError(
                            f"Array element at index {i} for datetime field '{field_name}' must be a valid datetime string. Got '{element}'. Use formats like: '2024-01-15', '2024-01-15T10:30:00', 'Jan 15, 2024'"
                        ) from e

        return tree

    def field_in_list(self, tree) -> Any:
        return self._validate_field_in_list_common(
            tree.children[0], tree.children[2], "in", tree
        )

    def field_not_in_list(self, tree) -> Any:
        return self._validate_field_in_list_common(
            tree.children[0], tree.children[2], "not in", tree
        )


class MilvusQueryTransformer(Transformer):
    """Transforms a validated tree into a Milvus-compatible query string."""

    def __init__(self, metadata_schema: MetadataSchema | None = None):
        self.metadata_schema = metadata_schema

    def start(self, args) -> str:
        result = str(args[0])
        return result

    def _check_null_contradictions_in_transformer(
        self, args: list[Any], clause_type: str
    ) -> None:
        """Check for contradictory IS NULL and IS NOT NULL clauses in transformer."""
        _check_null_contradictions_common(args, clause_type)

    def and_expr(self, args) -> str:
        self._check_null_contradictions_in_transformer(args, "AND")

        expressions = []
        for arg in args:
            if str(arg).lower() not in ["and", "or"]:
                expressions.append(str(arg))

        if len(expressions) == 1:
            return expressions[0]
        else:
            return " and ".join(expressions)

    def or_expr(self, args) -> str:
        self._check_null_contradictions_in_transformer(args, "OR")

        expressions = []
        for arg in args:
            if str(arg).lower() not in ["and", "or"]:
                expressions.append(str(arg))

        if len(expressions) == 1:
            return expressions[0]
        else:
            return " or ".join(expressions)

    def not_expr(self, args) -> str:
        if len(args) == 2:
            expr = str(args[1])
            return f"not ({expr})"
        elif len(args) == 1:
            expr = str(args[0])
            return f"not ({expr})"
        else:
            result = "not (" + " ".join(str(a) for a in args) + ")"
            return result

    def paren_expr(self, args) -> str:
        if len(args) == 3:
            expr = str(args[1])
        elif len(args) == 1:
            expr = str(args[0])
        else:
            expr = " ".join(str(a) for a in args)
        return f"({expr})"

    def comparison(self, args) -> str:
        field_val = str(args[0])
        op_val = str(args[1])
        value_token = args[2] if len(args) > 2 else None

        if value_token is not None:
            field_name = self._extract_field_name(field_val)
            if field_name and self.metadata_schema:
                field_info = self.metadata_schema.field_dict.get(field_name)
                if field_info and is_array_type(field_info.type):
                    if (
                        hasattr(value_token, "data")
                        and value_token.data == "array_literal"
                    ):
                        if (
                            len(value_token.children) == 3
                            and value_token.children[1] is None
                        ) or (
                            len(value_token.children) == 2
                            and hasattr(value_token.children[0], "type")
                            and value_token.children[0].type == "LBRACK"
                            and hasattr(value_token.children[1], "type")
                            and value_token.children[1].type == "RBRACK"
                        ):
                            raise FilterSemanticError(
                                f"Empty array comparisons are not supported for field '{field_val}'. "
                                f"Please provide non-empty array values or use explicit field comparisons."
                            )
                    elif str(value_token).strip() == "[]":
                        raise FilterSemanticError(
                            f"Empty array comparisons are not supported for field '{field_val}'. "
                            f"Please provide non-empty array values or use explicit field comparisons."
                        )

        if value_token is not None:
            field_name = self._extract_field_name(field_val)
            if field_name and self.metadata_schema:
                field_info = self.metadata_schema.field_dict.get(field_name)
                if field_info and is_boolean_type(field_info.type):
                    value_str = str(value_token)
                    if value_str in BOOLEAN_VALUES:
                        value_val = str(BOOLEAN_VALUES[value_str]).lower()
                    else:
                        value_val = str(value_token)
                elif field_info and is_datetime_type(field_info.type):
                    value_str = str(value_token)
                    try:
                        normalized_datetime = DatetimeUtility.parse_datetime(
                            value_str, context="filter", operator=op_val
                        )
                        value_val = f'"{normalized_datetime}"'

                        if op_val in [
                            "==",
                            "=",
                        ] and not DatetimeUtility._has_explicit_time(value_str):
                            (
                                start_str,
                                end_str,
                            ) = DatetimeUtility.convert_date_equality_to_between(
                                value_str
                            )
                            start_val = f'"{start_str}"'
                            end_val = f'"{end_str}"'
                            return f"({field_val} >= {start_val} and {field_val} <= {end_val})"

                    except (ValueError, TypeError) as e:
                        logger.debug(f"[comparison] Failed to normalize datetime: {e}")
                        value_val = str(value_token)
                elif field_info and is_string_type(field_info.type):
                    value_val = str(value_token).lower()
                else:
                    value_val = str(value_token)
            else:
                value_val = str(value_token)
        else:
            value_val = ""

        result = f"{field_val} {op_val} {value_val}"
        return result

    def like_comparison(self, args) -> str:
        field_val = str(args[0])
        op_val = str(args[1])
        pattern_val = str(args[2])

        return f"{field_val} {op_val} {pattern_val}"

    def between_comparison(self, args) -> str:
        field_val = str(args[0])
        start_val = str(args[2])
        end_val = str(args[4])
        field_name = self._extract_field_name(field_val)
        if field_name and self.metadata_schema:
            field_info = self.metadata_schema.field_dict.get(field_name)
            if field_info and is_datetime_type(field_info.type):
                try:
                    start_str = DatetimeUtility.parse_datetime(
                        start_val, context="filter", operator=">="
                    )
                    start_val = f'"{start_str}"'
                except (ValueError, TypeError) as e:
                    logger.debug(
                        f"[between_comparison] Failed to normalize start_val: {e}"
                    )

                try:
                    end_str = DatetimeUtility.parse_datetime(
                        end_val, context="filter", operator="<="
                    )
                    end_val = f'"{end_str}"'
                except (ValueError, TypeError) as e:
                    logger.debug(
                        f"[between_comparison] Failed to normalize end_val: {e}"
                    )

        return f"({field_val} >= {start_val} and {field_val} <= {end_val})"

    def is_null_comparison(self, args) -> str:
        """Convert IS NULL expressions to Milvus-compatible format."""
        field_val = str(args[0])
        op_val = str(args[1]).lower()

        raise FilterSemanticError(
            f"NULL operations (IS NULL, IS NOT NULL) are not supported in JSON metadata fields. "
            f"Field '{field_val}' cannot use '{op_val}'. "
            f"Please provide explicit values for all fields."
        )

    def before_after_comparison(self, args) -> str:
        field_val = str(args[0])
        op_val = str(args[1])
        value_val = str(args[2])
        if op_val.lower() == "before":
            op_val = "<"
        elif op_val.lower() == "after":
            op_val = ">"
        field_name = self._extract_field_name(field_val)
        if field_name and self.metadata_schema:
            field_info = self.metadata_schema.field_dict.get(field_name)
            if field_info and is_datetime_type(field_info.type):
                try:
                    normalized_datetime = DatetimeUtility.parse_datetime(
                        value_val, context="filter", operator=op_val
                    )
                    value_val = f'"{normalized_datetime}"'
                except (ValueError, TypeError) as e:
                    logger.debug(
                        f"[before_after_comparison] Failed to normalize datetime: {e}"
                    )

        return f"{field_val} {op_val} {value_val}"

    def _extract_field_name(self, field_val: str) -> str | None:
        """Extract field name from field reference for tracking."""
        match = re.match(r'content_metadata\["([^"]+)"\]', field_val)
        return match.group(1) if match else None

    def array_comparison(self, args) -> str:
        field_val = str(args[0])
        op_val = str(args[1])
        value_val = str(args[2])
        op_lower = op_val.lower()

        if value_val.strip() == "[]":
            raise FilterSemanticError(
                f"Empty array comparisons are not supported for field '{field_val}'. "
                f"Please provide non-empty array values or use explicit field comparisons."
            )

        field_name = self._extract_field_name(field_val)

        is_array_field = False
        if self.metadata_schema and field_name:
            field_info = self.metadata_schema.field_dict.get(field_name)
            if field_info and is_array_type(field_info.type):
                is_array_field = True

        if "includes" in op_lower and "not" not in op_lower:
            if value_val.startswith("["):
                return f"array_contains_all({field_val}, {value_val})"
            else:
                return f"array_contains({field_val}, {value_val})"
        elif "does not include" in op_lower or (
            "not" in op_lower and "includes" in op_lower
        ):
            if value_val.startswith("["):
                return f"not array_contains_any({field_val}, {value_val})"
            else:
                return f"not array_contains({field_val}, {value_val})"
        elif op_lower == "in":
            if is_array_field:
                if value_val.startswith("["):
                    return f"array_contains_any({field_val}, {value_val})"
                else:
                    return f"array_contains({field_val}, {value_val})"
            else:
                return f"{field_val} {op_val} {value_val}"
        elif op_lower == "not in":
            if is_array_field:
                if value_val.startswith("["):
                    return f"not array_contains_any({field_val}, {value_val})"
                else:
                    return f"not array_contains({field_val}, {value_val})"
            else:
                return f"{field_val} {op_val} {value_val}"
        else:
            return f"{field_val} {op_val} {value_val}"

    def array_function(self, args) -> str:
        func_name = str(args[0])
        if func_name.lower() == "array_length":
            field_val = str(args[2])
            return f"array_length({field_val})"
        else:
            field_val = str(args[2])
            value_val = str(args[4])

            if value_val.strip() == "[]":
                raise FilterSemanticError(
                    f"Empty array comparisons are not supported for field '{field_val}'. "
                    f"Please provide non-empty array values or use explicit field comparisons."
                )

            return f"{func_name}({field_val}, {value_val})"

    def array_length_comparison(self, args) -> str:
        field_val = str(args[2])
        op_val = str(args[4])
        value_val = str(args[5])

        if op_val == "==" and value_val == "0":
            raise FilterSemanticError(
                f"NULL operations (array_length == 0) are not supported for field '{field_val}'. "
                f"Please use explicit array comparisons instead."
            )
        elif op_val == "!=" and value_val == "0":
            raise FilterSemanticError(
                f"NULL operations (array_length != 0) are not supported for field '{field_val}'. "
                f"Please use explicit array comparisons instead."
            )
        else:
            return f"array_length({field_val}) {op_val} {value_val}"

    def array_membership(self, args) -> str:
        value_val = str(args[0])
        field_val = str(args[2])

        if value_val.strip() == "[]":
            raise FilterSemanticError(
                f"Empty array comparisons are not supported for field '{field_val}'. "
                f"Please provide non-empty array values or use explicit field comparisons."
            )

        return f"array_contains({field_val}, {value_val})"

    def array_membership_negated(self, args) -> str:
        value_val = str(args[0])
        field_val = str(args[2])

        if value_val.strip() == "[]":
            raise FilterSemanticError(
                f"Empty array comparisons are not supported for field '{field_val}'. "
                f"Please provide non-empty array values or use explicit field comparisons."
            )

        return f"not array_contains({field_val}, {value_val})"

    def field(self, args) -> str:
        return str(args[0])

    def value(self, args) -> str:
        if isinstance(args[0], list):
            return f"[{', '.join(args[0])}]"
        return str(args[0])

    def ESCAPED_STRING(self, token) -> str:
        """Convert single quotes to double quotes for Milvus compatibility."""
        value = token.value
        if value.startswith("'") and value.endswith("'"):
            return f'"{value[1:-1].lower()}"'
        if value == '""':
            raise FilterSemanticError(
                "Empty string values are not supported in filter expressions. "
                "Please provide non-empty string values."
            )
        return value

    def SIGNED_NUMBER(self, token) -> object:
        value_str = str(token.value)
        try:
            if "." in value_str:
                return float(value_str)
            else:
                return int(value_str)
        except (ValueError, TypeError):
            return token.value

    def BOOLEAN(self, token) -> bool:
        val = str(token.value).lower()
        if val in BOOLEAN_VALUES:
            return BOOLEAN_VALUES[val]
        return token.value

    def NULL_VALUE(self, token) -> str:
        """Reject NULL values as they are not supported in filter expressions."""
        raise FilterSemanticError(
            f"NULL values are not supported in filter expressions. "
            f"Got '{token.value}'. Please provide explicit values for all fields."
        )

    def array_literal(self, args) -> str:
        if len(args) < 3 or args[1] is None:
            return "[]"
        elements = args[1] if isinstance(args[1], list) else [args[1]]
        primitive_elements = []
        for el in elements:
            if isinstance(el, str):
                if el == "":
                    continue
                else:
                    primitive_elements.append(f'"{el.lower()}"')
            elif isinstance(el, bool):
                primitive_elements.append(str(el).lower())
            elif isinstance(el, int | float):
                primitive_elements.append(el)
            else:
                primitive_elements.append(str(el))
        return f"[{', '.join(str(e) for e in primitive_elements)}]"

    def array_elements(self, args) -> list:
        elements = []
        for arg in args:
            if isinstance(arg, list):
                filtered_elements = [el for el in arg if el is not None]
                elements.extend(filtered_elements)
            elif not (isinstance(arg, str) and arg == ","):
                if arg is not None:
                    elements.append(arg)
        return elements

    def array_element(self, args) -> object:
        def extract_primitive(el):
            while isinstance(el, Tree) and hasattr(el, "children") and el.children:
                el = el.children[0]
            if isinstance(el, Token):
                if el.type == "SIGNED_NUMBER":
                    return el.value
                elif el.type == "BOOLEAN":
                    val = str(el.value).lower()
                    if val in BOOLEAN_VALUES:
                        return BOOLEAN_VALUES[val]
                    return el.value
                elif el.type == "ESCAPED_STRING":
                    value = str(el.value)[1:-1]
                    if value == "":
                        return None
                    return value.lower()
                else:
                    return el.value
            elif hasattr(el, "value"):
                return el.value
            if isinstance(el, str):
                s = el.strip("\"'")
                if s == "":
                    return None
                return s.lower()
            return el

        result = extract_primitive(args[0])
        return result

    def LIKE_OP(self, token) -> str:
        return token.value

    def COMPARISON_OP(self, token) -> str:
        """Convert = to == for Milvus compatibility."""
        value = token.value
        if value == "=":
            return "=="
        return value

    def BETWEEN_KEYWORD(self, token) -> str:
        return token.value

    def BEFORE_AFTER_OP(self, token) -> str:
        return token.value

    def ARRAY_OP(self, token) -> str:
        return token.value

    def ARRAY_FUNC(self, token) -> str:
        return token.value

    def IN_OP(self, token) -> str:
        return token.value

    def LOGICAL_AND(self, token) -> str:
        return "and"

    def LOGICAL_OR(self, token) -> str:
        return "or"

    def LOGICAL_NOT(self, token) -> str:
        return "not"

    def LPAREN(self, token) -> str:
        return "("

    def RPAREN(self, token) -> str:
        return ")"

    def LBRACK(self, token) -> str:
        return "["

    def RBRACK(self, token) -> str:
        return "]"

    def COMMA(self, token) -> str:
        return ","

    def field_in_list(self, args) -> str:
        field_val = str(args[0])
        value_val = str(args[2])

        if value_val.strip() == "[]":
            raise FilterSemanticError(
                f"Empty array comparisons are not supported for field '{field_val}'. "
                f"Please provide non-empty array values or use explicit field comparisons."
            )

        field_name = self._extract_field_name(field_val)

        if self.metadata_schema and field_name:
            field_info = self.metadata_schema.field_dict.get(field_name)
            if field_info and is_array_type(field_info.type):
                return f"array_contains_any({field_val}, {value_val})"
        return f"{field_val} in {value_val}"

    def field_not_in_list(self, args) -> str:
        field_val = str(args[0])
        value_val = str(args[2])

        if value_val.strip() == "[]":
            raise FilterSemanticError(
                f"Empty array comparisons are not supported for field '{field_val}'. "
                f"Please provide non-empty array values or use explicit field comparisons."
            )

        field_name = self._extract_field_name(field_val)

        if self.metadata_schema and field_name:
            field_info = self.metadata_schema.field_dict.get(field_name)
            if field_info and is_array_type(field_info.type):
                return f"not array_contains_any({field_val}, {value_val})"
        return f"{field_val} not in {value_val}"


class FilterExpressionParser:
    """Parser for filter expressions with semantic validation."""

    def __init__(self, metadata_schema: MetadataSchema, config):
        self.metadata_schema = metadata_schema
        self.config = validate_metadata_config(config)
        self.schema_dict = metadata_schema.field_dict
        self.parser = get_grammar_parser()
        self.validator = FilterSemanticValidator(
            metadata_schema=metadata_schema, config=config
        )
        self.transformer = MilvusQueryTransformer(metadata_schema=metadata_schema)

    def _parse_and_validate(self, filter_expr: str) -> tuple[Any, str]:
        """Parse and validate a filter expression."""
        if not filter_expr or (
            isinstance(filter_expr, str) and filter_expr.strip() == ""
        ):
            return None, ""

        try:
            tree = self.parser.parse(filter_expr)
            self.validator.visit(tree)
            processed_expr = self.transformer.transform(tree)
            return tree, processed_expr
        except UnexpectedInput as e:
            error_context = self._get_error_context(filter_expr, e)
            raise FilterSyntaxError(
                f"Invalid filter syntax: {error_context}\n\n"
                f"Please check your filter expression and ensure it uses supported syntax. "
                f"Complex expressions or unsupported literals may need to be simplified."
            ) from e
        except (FilterSemanticError, FilterSyntaxError):
            raise
        except Exception as e:
            raise FilterSyntaxError(
                f"Error processing filter expression: {str(e)}\n\n"
                f"Please check the syntax and ensure all values are in supported formats."
            ) from e

    def validate_filter_expression(self, filter_expr: str) -> FilterExpressionResult:
        """Validate a filter expression for syntax and semantics."""
        try:
            self._parse_and_validate(filter_expr)
            return {
                "status": True,
                "message": "Filter expression validated successfully",
            }
        except FilterSyntaxError as e:
            return {"status": False, "error_message": str(e)}
        except FilterSemanticError as e:
            return {"status": False, "error_message": str(e)}

    def process_filter_expression(self, filter_expr: str) -> dict[str, Any]:
        """Parse, validate, and transform a filter expression to Milvus format."""
        try:
            tree, processed_expr = self._parse_and_validate(filter_expr)
            if tree is None:
                return {
                    "status": True,
                    "processed_expression": "",
                    "message": "Empty filter expression",
                }
            return {
                "status": True,
                "processed_expression": processed_expr,
                "message": "Filter expression processed successfully",
            }
        except FilterSyntaxError as e:
            return {"status": False, "error_message": str(e)}
        except FilterSemanticError as e:
            return {"status": False, "error_message": str(e)}

    @staticmethod
    def _get_error_context(filter_expr: str, error: UnexpectedInput) -> str:
        """Get helpful error context for parsing errors."""
        if not isinstance(filter_expr, str):
            return (
                f"Invalid filter expression type: {type(filter_expr)}. Expected string."
            )

        pos = getattr(error, "pos_in_stream", None)
        if pos is not None:
            lines = filter_expr[:pos].split("\n")
            line_num = len(lines)
            col_num = len(lines[-1]) if lines else 0

            start = max(0, pos - 20)
            end = min(len(filter_expr), pos + 20)
            snippet = filter_expr[start:end]

            error_msg = (
                f"Syntax error at line {line_num}, column {col_num}: '{snippet}'"
            )
            error_msg += "\n\nExamples of valid filter expressions:"
            error_msg += "\n• content_metadata[\"title\"] == 'value'"
            error_msg += "\n• content_metadata[\"title\"] = 'value'"
            error_msg += '\n• content_metadata["rating"] > 5'
            error_msg += "\n• content_metadata[\"category\"] like '%tech%'"
            error_msg += "\n• content_metadata[\"tags\"] in ['important', 'urgent']"
            error_msg += "\n• content_metadata[\"created_date\"] between '2024-01-01' and '2024-12-31'"
            error_msg += '\n• content_metadata["is_public"] == true'
            error_msg += '\n• content_metadata["file_size"] > 1000 and content_metadata["type"] == \'pdf\''

            return error_msg

        return str(error)


class MetadataValidator:
    """Validates metadata values against schema."""

    def __init__(self, config) -> None:
        self.config = validate_metadata_config(config)
        self.max_array_length = self.config.max_array_length
        self.max_string_length = self.config.max_string_length

    def validate_and_normalize_metadata_values(
        self, metadata: MetadataDict, schema: MetadataSchema
    ) -> tuple[bool, list[dict[str, Any]], MetadataDict]:
        """Validate and normalize metadata values against schema using Pydantic."""
        try:
            model_class = get_cached_model_class(schema, self.config)

            model_instance = model_class(**metadata)

            normalized_data = model_instance.model_dump()

            logger.debug("Metadata values validated and normalized successfully.")
            return True, [], normalized_data

        except ValidationError as e:
            errors = []
            for error in e.errors():
                field_name = error["loc"][0] if error["loc"] else "unknown"
                error_msg = error["msg"]
                errors.append({"error": f"Field '{field_name}': {error_msg}"})
            logger.error(f"Metadata value validation errors: {errors}")
            return False, errors, metadata
        except Exception as e:
            logger.error(f"Unexpected error during metadata validation: {e}")
            return (
                False,
                [{"error": f"Unexpected error during validation: {str(e)}"}],
                metadata,
            )


__all__ = [
    # Core classes
    "MetadataField",
    "MetadataSchema",
    "MetadataValidator",
    "FilterExpressionParser",
    "FilterSemanticValidator",
    "MilvusQueryTransformer",
    "DatetimeUtility",
    # New validator classes
    "BaseValidator",
    "ValueValidator",
    "OperatorValidator",
    "FieldValidator",
    "ArrayValidator",
    # Exceptions
    "MetadataValidationError",
    "FilterSyntaxError",
    "FilterSemanticError",
    "MetadataConfigError",
    # Result types
    "FilterExpressionResult",
    "MetadataDict",
    "MetadataValue",
    "get_valid_array_types",
    "validate_metadata_config",
    # Type mapping constants
    "METADATA_TYPE_TO_PYTHON_TYPE",
    "ARRAY_TYPE_TO_PYTHON_TYPE",
    "VALID_ARRAY_TYPES_AS_STRINGS",
    # Type helper functions
    "is_string_type",
    "is_datetime_type",
    "is_numeric_type",
    "is_array_type",
    "is_boolean_type",
    "get_python_type_for_metadata_type",
    # Caching functions
    "get_grammar_parser",
    "get_cached_model_class",
]

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

"""Schema-aware validator and processor for Elasticsearch DSL filter clauses.

Mirrors the Milvus FilterExpressionParser interface so that callers in
`utils/common.py` can invoke the same validate/process surface area for both
backends. Operates on Elasticsearch Query DSL clauses (already structured JSON)
rather than parsing a textual grammar.
"""

import copy
import logging
from typing import Any

from dateutil import parser as dt_parser

from nvidia_rag.utils.metadata_validation import (
    SYSTEM_MANAGED_FIELDS,
    FilterSemanticError,
    FilterSyntaxError,
    MetadataField,
    MetadataSchema,
    MetadataType,
    validate_metadata_config,
)

logger = logging.getLogger(__name__)


_METADATA_PREFIX = "metadata.content_metadata."
_KEYWORD_SUFFIX = ".keyword"

_LEAF_CLAUSE_TYPES = {
    "term",
    "terms",
    "range",
    "match",
    "match_phrase",
    "wildcard",
    "prefix",
    "exists",
}
_BOOL_CLAUSE_TYPES = {"bool"}
_BOOL_INNER_KEYS = {"must", "should", "must_not", "filter"}

# Per-field-type to compatible-clause-type mapping. Mirrors
# `TYPE_OPERATOR_MAPPING` in metadata_validation.py but in ES vocabulary.
_TYPE_TO_CLAUSES: dict[str, set[str]] = {
    MetadataType.STRING.value: {
        "term",
        "terms",
        "match",
        "match_phrase",
        "wildcard",
        "prefix",
        "exists",
    },
    MetadataType.DATETIME.value: {"term", "terms", "range", "exists"},
    MetadataType.NUMBER.value: {"term", "terms", "range", "exists"},
    MetadataType.INTEGER.value: {"term", "terms", "range", "exists"},
    MetadataType.FLOAT.value: {"term", "terms", "range", "exists"},
    MetadataType.BOOLEAN.value: {"term", "terms", "exists"},
    MetadataType.ARRAY.value: {"term", "terms", "exists"},
}

_RANGE_OPERATORS = {"gt", "gte", "lt", "lte", "from", "to"}

# Hard caps to prevent abuse. Kept conservative; configurable via
# MetadataConfig in a follow-up if needed.
_MAX_DEPTH = 8
_MAX_CLAUSES = 64


class ElasticsearchFilterValidator:
    """Schema-aware validator for Elasticsearch DSL filter clauses."""

    def __init__(self, metadata_schema: MetadataSchema, config: Any) -> None:
        self.metadata_schema = metadata_schema
        self.config = validate_metadata_config(config)
        self.schema_dict: dict[str, MetadataField] = metadata_schema.field_dict

    # ------------------------------------------------------------------
    # Public API — mirrors FilterExpressionParser shape
    # ------------------------------------------------------------------

    def validate_filter(self, filter_clauses: list[dict[str, Any]]) -> dict[str, Any]:
        """Validate a list of ES DSL clauses against the metadata schema."""
        try:
            self._walk_clauses(filter_clauses, normalize=False)
            return {
                "status": True,
                "message": "Filter clauses validated successfully",
            }
        except (FilterSyntaxError, FilterSemanticError) as e:
            return {"status": False, "error_message": str(e)}

    def process_filter(self, filter_clauses: list[dict[str, Any]]) -> dict[str, Any]:
        """Validate and normalize ES DSL clauses (e.g. add `.keyword` where needed).

        Emits a WARNING listing every field-path change. Logged at WARNING
        (not INFO) because we are silently rewriting user-supplied input —
        the caller should fix the source so the rewrite eventually goes
        away.
        """
        try:
            normalized = copy.deepcopy(filter_clauses)
            changes: list[tuple[str, str]] = []
            self._walk_clauses(normalized, normalize=True, changes=changes)
            if changes:
                rewrites = ", ".join(f"{old} -> {new}" for old, new in changes)
                logger.warning(
                    "[ES Filter] Rewrote %d path(s): %s "
                    "(`.keyword` is only valid on string fields).",
                    len(changes),
                    rewrites,
                )
            return {
                "status": True,
                "processed_expression": normalized,
                "message": "Filter clauses processed successfully",
                "normalization_changes": changes,
            }
        except (FilterSyntaxError, FilterSemanticError) as e:
            return {"status": False, "error_message": str(e)}

    # ------------------------------------------------------------------
    # Walker
    # ------------------------------------------------------------------

    def _walk_clauses(
        self,
        clauses: list[dict[str, Any]],
        *,
        normalize: bool,
        depth: int = 0,
        clause_count: list[int] | None = None,
        changes: list[tuple[str, str]] | None = None,
    ) -> None:
        if clause_count is None:
            clause_count = [0]

        if not isinstance(clauses, list):
            raise FilterSyntaxError(
                "Elasticsearch filter must be a JSON array of clause objects."
            )

        if depth > _MAX_DEPTH:
            raise FilterSyntaxError(
                f"Filter clause nesting depth exceeds maximum ({_MAX_DEPTH})."
            )

        for clause in clauses:
            clause_count[0] += 1
            if clause_count[0] > _MAX_CLAUSES:
                raise FilterSyntaxError(
                    f"Filter contains more than {_MAX_CLAUSES} clauses."
                )
            self._walk_clause(
                clause,
                normalize=normalize,
                depth=depth,
                clause_count=clause_count,
                changes=changes,
            )

    def _walk_clause(
        self,
        clause: dict[str, Any],
        *,
        normalize: bool,
        depth: int,
        clause_count: list[int],
        changes: list[tuple[str, str]] | None = None,
    ) -> None:
        if not isinstance(clause, dict) or len(clause) != 1:
            raise FilterSyntaxError(
                "Each filter clause must be a dict with exactly one top-level key "
                "(e.g. 'term', 'range', 'bool')."
            )

        clause_type, body = next(iter(clause.items()))

        if clause_type in _BOOL_CLAUSE_TYPES:
            self._walk_bool(
                body,
                normalize=normalize,
                depth=depth + 1,
                clause_count=clause_count,
                changes=changes,
            )
            return

        if clause_type not in _LEAF_CLAUSE_TYPES:
            raise FilterSyntaxError(
                f"Unsupported clause type '{clause_type}'. Supported: "
                f"{sorted(_LEAF_CLAUSE_TYPES | _BOOL_CLAUSE_TYPES)}."
            )

        self._validate_leaf(
            clause, clause_type, body, normalize=normalize, changes=changes
        )

    def _walk_bool(
        self,
        body: Any,
        *,
        normalize: bool,
        depth: int,
        clause_count: list[int],
        changes: list[tuple[str, str]] | None = None,
    ) -> None:
        if not isinstance(body, dict):
            raise FilterSyntaxError(
                "'bool' clause body must be an object containing must/should/must_not/filter."
            )

        unknown = (
            set(body.keys()) - _BOOL_INNER_KEYS - {"minimum_should_match", "boost"}
        )
        if unknown:
            raise FilterSyntaxError(
                f"Unsupported keys in 'bool' clause: {sorted(unknown)}. "
                f"Supported: {sorted(_BOOL_INNER_KEYS)}."
            )

        for key in _BOOL_INNER_KEYS:
            inner = body.get(key)
            if inner is None:
                continue
            # ES allows single-clause shorthand (dict) or list — normalize to list for walking.
            if isinstance(inner, dict):
                inner_list = [inner]
                if normalize:
                    body[key] = inner_list
            elif isinstance(inner, list):
                inner_list = inner
            else:
                raise FilterSyntaxError(
                    f"'bool.{key}' must be an object or array of clause objects."
                )
            self._walk_clauses(
                inner_list,
                normalize=normalize,
                depth=depth,
                clause_count=clause_count,
                changes=changes,
            )

    # ------------------------------------------------------------------
    # Leaf clause validation
    # ------------------------------------------------------------------

    def _validate_leaf(
        self,
        clause: dict[str, Any],
        clause_type: str,
        body: Any,
        *,
        normalize: bool,
        changes: list[tuple[str, str]] | None = None,
    ) -> None:
        # `exists` body is `{"field": "<path>"}`; others are `{"<path>": <value>}`.
        if clause_type == "exists":
            if not isinstance(body, dict) or "field" not in body:
                raise FilterSyntaxError(
                    "'exists' clause body must contain a 'field' key."
                )
            field_path = body["field"]
            if not isinstance(field_path, str):
                raise FilterSyntaxError(
                    "'exists' clause 'field' value must be a string."
                )
            field_name, had_keyword = self._strip_field_path(field_path)
            field_def = self._lookup_field(field_name)
            self._assert_clause_compatible(field_def, clause_type)
            if normalize:
                normalized_path = self._normalize_field_path(
                    field_def, clause_type, field_name, had_keyword
                )
                if normalized_path != field_path:
                    clause[clause_type] = {"field": normalized_path}
                    if changes is not None:
                        changes.append((field_path, normalized_path))
            return

        if not isinstance(body, dict) or len(body) != 1:
            raise FilterSyntaxError(
                f"'{clause_type}' clause body must be a dict with exactly one field path."
            )

        field_path, value = next(iter(body.items()))
        if not isinstance(field_path, str):
            raise FilterSyntaxError(
                f"'{clause_type}' clause field path must be a string."
            )

        field_name, had_keyword = self._strip_field_path(field_path)
        field_def = self._lookup_field(field_name)
        self._assert_clause_compatible(field_def, clause_type)

        if clause_type == "range":
            self._validate_range_value(field_def, value)
        elif clause_type == "terms":
            if not isinstance(value, list):
                raise FilterSyntaxError("'terms' clause value must be an array.")

        # Normalization: ensure `.keyword` suffix where appropriate, and strip
        # it from non-string-like fields where it would target a non-existent
        # ES sub-field.
        if normalize:
            normalized_path = self._normalize_field_path(
                field_def, clause_type, field_name, had_keyword
            )
            if normalized_path != field_path:
                clause[clause_type] = {normalized_path: value}
                if changes is not None:
                    changes.append((field_path, normalized_path))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _strip_field_path(self, field_path: str) -> tuple[str, bool]:
        """Return (bare_field_name, had_keyword_suffix)."""
        path = field_path
        if path.startswith(_METADATA_PREFIX):
            path = path[len(_METADATA_PREFIX) :]
        had_keyword = False
        if path.endswith(_KEYWORD_SUFFIX):
            had_keyword = True
            path = path[: -len(_KEYWORD_SUFFIX)]
        # Strip any further dotted suffix segments (e.g. multi-field analyzers).
        # The first segment is the metadata field name we registered.
        bare = path.split(".", 1)[0]
        return bare, had_keyword

    def _lookup_field(self, field_name: str) -> MetadataField:
        if field_name in self.schema_dict:
            return self.schema_dict[field_name]

        sys_def = SYSTEM_MANAGED_FIELDS.get(field_name)
        if sys_def and sys_def.get("support_dynamic_filtering", False):
            return MetadataField(
                name=field_name,
                type=sys_def["type"],
                user_defined=False,
                support_dynamic_filtering=True,
                array_type="string"
                if sys_def["type"] == MetadataType.ARRAY.value
                else None,
            )

        raise FilterSemanticError(
            f"Field '{field_name}' is not defined in the metadata schema and is not "
            f"a dynamically-filterable system field. Available fields: "
            f"{sorted(self.schema_dict.keys())}."
        )

    def _assert_clause_compatible(
        self, field_def: MetadataField, clause_type: str
    ) -> None:
        allowed = _TYPE_TO_CLAUSES.get(field_def.type, set())
        if clause_type not in allowed:
            raise FilterSemanticError(
                f"Clause '{clause_type}' is not compatible with field "
                f"'{field_def.name}' of type '{field_def.type}'. "
                f"Allowed clauses: {sorted(allowed)}."
            )

    def _validate_range_value(self, field_def: MetadataField, value: Any) -> None:
        if not isinstance(value, dict) or not value:
            raise FilterSyntaxError(
                "'range' clause value must be a non-empty object with operators "
                "(gt, gte, lt, lte)."
            )
        unknown = set(value.keys()) - _RANGE_OPERATORS - {"format", "time_zone"}
        if unknown:
            raise FilterSyntaxError(
                f"Unsupported range operators: {sorted(unknown)}. "
                f"Supported: {sorted(_RANGE_OPERATORS)}."
            )
        if field_def.type == MetadataType.DATETIME.value:
            for op in _RANGE_OPERATORS & set(value.keys()):
                v = value[op]
                if not isinstance(v, str):
                    raise FilterSyntaxError(
                        f"Datetime range bound '{op}' must be an ISO 8601 string."
                    )
                try:
                    dt_parser.isoparse(v)
                except (ValueError, TypeError) as e:
                    raise FilterSyntaxError(
                        f"Invalid ISO 8601 datetime for range '{op}': {v!r}"
                    ) from e

    def _normalize_field_path(
        self,
        field_def: MetadataField,
        clause_type: str,
        field_name: str,
        had_keyword: bool,
    ) -> str:
        """Produce the canonical ES field path for a (clause, field) pair.

        - Always prefixes with `metadata.content_metadata.`.
        - Adds `.keyword` for exact-match clauses on string-like fields.
        - Strips a caller-supplied `.keyword` for non-string-like fields.
          Rationale: `.keyword` is an Elasticsearch text multi-field — it only
          exists on `string`-typed fields. Targeting `<numeric_field>.keyword`
          points at a non-existent mapping and silently returns zero hits
          (which previously broke UI-issued filters such as
          `term: {priority.keyword: 2}` on an integer `priority`).
        """
        path = f"{_METADATA_PREFIX}{field_name}"

        is_string_like = field_def.type == MetadataType.STRING.value or (
            field_def.type == MetadataType.ARRAY.value
            and field_def.array_type == MetadataType.STRING.value
        )
        # `term`/`terms`/`prefix` need `.keyword` on string-like fields to do
        # exact (non-analyzed) matching. Other clause types either don't apply
        # to strings or work on the analyzed text directly.
        wants_keyword = clause_type in {"term", "terms", "prefix"} and is_string_like

        if wants_keyword:
            path = f"{path}{_KEYWORD_SUFFIX}"
        elif had_keyword and is_string_like:
            # Preserve user intent for string-like fields with non-default
            # clauses (e.g. `wildcard` / `match` on the keyword sub-field).
            path = f"{path}{_KEYWORD_SUFFIX}"
        # else: caller supplied `.keyword` on a non-string field — drop it.

        return path

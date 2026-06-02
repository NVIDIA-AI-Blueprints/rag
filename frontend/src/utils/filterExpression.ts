// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import type { Filter } from "../types/chat";

/**
 * Vector-store backends the UI knows how to serialize filters for. Detected
 * from `/health.databases[].service` (Elasticsearch | Milvus).
 */
export type VectorStoreKind = "milvus" | "elasticsearch";

/** Per-field type info needed to format values correctly (quoting, etc.). */
export interface FieldTypeMap {
  /** field name -> type ("string" | "integer" | "float" | "number" | "boolean" | "datetime" | "array") */
  fieldType: Map<string, string>;
  /** field name -> array element type (set only when fieldType === "array") */
  arrayElementType: Map<string, string>;
}

/** Pre-build field type maps from the selected collections' metadata schema. */
export const buildFieldTypeMap = (
  schemas: Array<
    Array<{
      name: string;
      type: string;
      array_type?: string | null | undefined;
    }>
  >
): FieldTypeMap => {
  const fieldType = new Map<string, string>();
  const arrayElementType = new Map<string, string>();
  for (const schema of schemas) {
    for (const field of schema) {
      fieldType.set(field.name, field.type);
      if (field.array_type) {
        arrayElementType.set(field.name, field.array_type);
      }
    }
  }
  return { fieldType, arrayElementType };
};

/**
 * Identify a backend from the `/health.databases[].service` string. The BE
 * uses the human-readable label ("Elasticsearch" / "Milvus"), so we
 * normalize case-insensitively.
 */
export const vectorStoreFromHealthService = (
  service: string | undefined
): VectorStoreKind | undefined => {
  if (!service) return undefined;
  const lower = service.toLowerCase();
  if (lower.includes("elasticsearch") || lower === "es") return "elasticsearch";
  if (lower.includes("milvus")) return "milvus";
  return undefined;
};

// =============================================================================
// MILVUS FILTER COMPILER
// =============================================================================
// Milvus filter_expr is a single string expression using
// `content_metadata["field"] op value` syntax with AND/OR (flat, default
// precedence). See docs/custom-metadata.md "Filter Expression Syntax".

const formatMilvusValue = (
  value: Filter["value"],
  isArrayField: boolean,
  arrayElementType: string | undefined
): string => {
  if (Array.isArray(value)) {
    const items = value.map((item) => {
      if (typeof item === "boolean" || typeof item === "number") {
        return String(item);
      }
      // For array-typed fields, leave numeric/boolean array elements
      // unquoted — quoting them would break Milvus's type coercion.
      if (
        isArrayField &&
        arrayElementType &&
        ["integer", "float", "number", "boolean"].includes(arrayElementType)
      ) {
        return String(item);
      }
      return `"${item}"`;
    });
    return `[${items.join(", ")}]`;
  }
  if (typeof value === "boolean" || typeof value === "number") {
    return String(value);
  }
  return `"${value}"`;
};

const compileMilvusClause = (
  f: Filter,
  fieldTypes: FieldTypeMap
): string => {
  const isArrayField = fieldTypes.fieldType.get(f.field) === "array";
  const arrayElementType = fieldTypes.arrayElementType.get(f.field);
  const formatted = formatMilvusValue(f.value, isArrayField, arrayElementType);

  switch (f.operator) {
    case "array_contains":
    case "array_contains_all":
    case "array_contains_any":
      return `${f.operator}(content_metadata["${f.field}"], ${formatted})`;
    default:
      return `content_metadata["${f.field}"] ${f.operator} ${formatted}`;
  }
};

/**
 * Compile a list of UI filters to a Milvus `filter_expr` string. Returns
 * `undefined` for an empty input (so the caller can omit the field).
 *
 * Logical operators apply between clauses with default Milvus precedence
 * (no parenthesization), matching pre-existing behavior.
 */
export const compileMilvusFilter = (
  filters: Filter[],
  fieldTypes: FieldTypeMap
): string | undefined => {
  if (filters.length === 0) return undefined;
  return filters
    .map((f, index) => {
      const clause = compileMilvusClause(f, fieldTypes);
      if (index === 0) return clause;
      const op = (f.logicalOperator || "OR").toLowerCase();
      return ` ${op} ${clause}`;
    })
    .join("");
};

// =============================================================================
// ELASTICSEARCH FILTER COMPILER
// =============================================================================
// ES filter_expr is a list of dicts using Elasticsearch Query DSL. Field paths
// are `metadata.content_metadata.<name>`, with a `.keyword` suffix appended
// only for exact-match clauses (`term`/`terms`/`prefix`) on string-typed
// fields. Numeric, datetime, and boolean fields use the bare path because
// `.keyword` only exists as a multi-field on string mappings — appending it
// elsewhere targets a non-existent field and silently returns zero hits. See
// docs/custom-metadata.md#elasticsearch-filter-example.

/** ES Query DSL clause. Kept loose because ES accepts arbitrary shapes. */
export type ESClause = Record<string, unknown>;

/** Schema types that support a `.keyword` multi-field in ES mappings. */
const STRING_FIELD_TYPES = new Set(["string"]);

const isStringLikeField = (
  field: string,
  fieldTypes: FieldTypeMap
): boolean => {
  const t = fieldTypes.fieldType.get(field);
  if (!t) {
    // Unknown field (no schema loaded for the selected collection). Default
    // to string-like to preserve the previous wire format for installs
    // without registered schemas. The backend normalizer will strip
    // `.keyword` server-side if the field turns out to be non-string.
    return true;
  }
  if (STRING_FIELD_TYPES.has(t)) return true;
  if (t === "array") {
    const elem = fieldTypes.arrayElementType.get(field);
    return elem !== undefined && STRING_FIELD_TYPES.has(elem);
  }
  return false;
};

/**
 * Clause kinds we care about for the `.keyword` decision:
 * - "exact"   → `term` / `terms` / `prefix` / `wildcard` / `match`: append
 *   `.keyword` when the field is string-like (that's where the ES multi-field
 *   exists). For non-string fields these clauses either don't apply or work
 *   on the native field type, so use the bare path.
 * - "range"  → never `.keyword`. String fields don't support range queries
 *   server-side, and on numeric/datetime fields the keyword sub-field
 *   doesn't exist.
 */
type EsClauseKind = "exact" | "range";

const esField = (
  name: string,
  clauseKind: EsClauseKind,
  fieldTypes: FieldTypeMap
): string => {
  const base = `metadata.content_metadata.${name}`;
  if (clauseKind === "range") return base;
  return isStringLikeField(name, fieldTypes) ? `${base}.keyword` : base;
};

/** Convert Milvus-style `%` wildcards to ES `*`/`?`. */
const milvusLikeToEsWildcard = (pattern: string): string =>
  pattern.replace(/%/g, "*").replace(/_/g, "?");

const negate = (clause: ESClause): ESClause => ({
  bool: { must_not: [clause] },
});

const compileEsClause = (f: Filter, fieldTypes: FieldTypeMap): ESClause => {
  const exactPath = esField(f.field, "exact", fieldTypes);
  const rangePath = esField(f.field, "range", fieldTypes);

  switch (f.operator) {
    case "=":
    case "==":
      return { term: { [exactPath]: f.value } };

    case "!=":
      return negate({ term: { [exactPath]: f.value } });

    case ">":
      return { range: { [rangePath]: { gt: f.value } } };
    case ">=":
      return { range: { [rangePath]: { gte: f.value } } };
    case "<":
      return { range: { [rangePath]: { lt: f.value } } };
    case "<=":
      return { range: { [rangePath]: { lte: f.value } } };

    case "after":
      return { range: { [rangePath]: { gt: f.value } } };
    case "before":
      return { range: { [rangePath]: { lt: f.value } } };

    case "in":
      return {
        terms: { [exactPath]: Array.isArray(f.value) ? f.value : [f.value] },
      };
    case "not in":
      return negate({
        terms: { [exactPath]: Array.isArray(f.value) ? f.value : [f.value] },
      });

    case "like":
      return {
        wildcard: {
          [exactPath]: milvusLikeToEsWildcard(String(f.value)),
        },
      };

    // ES `term` already does set-membership for stored arrays, so a single
    // value matches any element of the array.
    case "array_contains":
      return { term: { [exactPath]: f.value } };

    // "all of" — emit one `term` per requested value, ANDed together.
    case "array_contains_all": {
      const values = Array.isArray(f.value) ? f.value : [f.value];
      return {
        bool: { must: values.map((v) => ({ term: { [exactPath]: v } })) },
      };
    }

    // "any of" — `terms` is a built-in OR-of-values match.
    case "array_contains_any":
    case "includes":
      return {
        terms: { [exactPath]: Array.isArray(f.value) ? f.value : [f.value] },
      };
    case "does not include":
      return negate({
        terms: { [exactPath]: Array.isArray(f.value) ? f.value : [f.value] },
      });

    default: {
      // Exhaustiveness guard — if the Filter operator union grows without
      // a corresponding case here, TypeScript will flag it. At runtime we
      // fall back to a `term` query so we still produce a valid request.
      const _exhaustive: never = f.operator;
      void _exhaustive;
      return { term: { [exactPath]: f.value } };
    }
  }
};

/**
 * Compile a list of UI filters to an Elasticsearch `filter_expr` (list of
 * dicts) per docs/custom-metadata.md.
 *
 * Grouping rules (match the Milvus flat-precedence behavior):
 *
 * - All-AND or single filter → flat list of clauses (matches the doc
 *   example: `[{term: ...}, {range: ...}]`).
 * - Any OR → group consecutive AND-joined clauses, then OR the groups
 *   together via a single top-level `bool.should` envelope.
 *
 * Returns `undefined` for empty input (so the caller can omit the field).
 *
 * `fieldTypes` carries the schema-derived type info needed to decide whether
 * `.keyword` should be appended for a given clause/field pair. When a field
 * is missing from `fieldTypes` (e.g. no schema is registered), we default to
 * the previous string-like behavior and rely on the backend normalizer to
 * strip `.keyword` if the field is actually non-string.
 */
export const compileElasticsearchFilter = (
  filters: Filter[],
  fieldTypes: FieldTypeMap = {
    fieldType: new Map(),
    arrayElementType: new Map(),
  }
): ESClause[] | undefined => {
  if (filters.length === 0) return undefined;

  // Split into AND-groups separated by OR boundaries. The first filter
  // always opens the first group. Subsequent filters extend the current
  // group on AND, or open a new group on OR.
  const groups: ESClause[][] = [[]];
  filters.forEach((f, index) => {
    const op = index === 0 ? "AND" : f.logicalOperator || "OR";
    if (op === "OR" && groups[groups.length - 1].length > 0) {
      groups.push([]);
    }
    groups[groups.length - 1].push(compileEsClause(f, fieldTypes));
  });

  if (groups.length === 1) {
    // All-AND or single → flat list. Matches the documented contract.
    return groups[0];
  }

  // Mixed / OR — wrap once in bool.should so the entire request still
  // satisfies the "list of dicts" shape (a single-element list).
  const shouldClauses = groups.map((g) =>
    g.length === 1 ? g[0] : { bool: { must: g } }
  );
  return [{ bool: { should: shouldClauses } }];
};

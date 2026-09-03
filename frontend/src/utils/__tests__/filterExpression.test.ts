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

import { describe, it, expect } from "vitest";
import type { Filter } from "../../types/chat";
import {
  buildFieldTypeMap,
  compileElasticsearchFilter,
  compileMilvusFilter,
  vectorStoreFromHealthService,
} from "../filterExpression";

const emptyTypes = buildFieldTypeMap([]);

describe("vectorStoreFromHealthService", () => {
  it.each([
    ["Elasticsearch", "elasticsearch"],
    ["elasticsearch", "elasticsearch"],
    ["ES", "elasticsearch"],
    ["Milvus", "milvus"],
    ["milvus", "milvus"],
  ])("normalizes %s to %s", (label, expected) => {
    expect(vectorStoreFromHealthService(label)).toBe(expected);
  });

  it("returns undefined for unrecognized labels", () => {
    expect(vectorStoreFromHealthService(undefined)).toBeUndefined();
    expect(vectorStoreFromHealthService("")).toBeUndefined();
    expect(vectorStoreFromHealthService("postgres")).toBeUndefined();
  });
});

describe("buildFieldTypeMap", () => {
  it("flattens schemas across multiple collections, last write wins", () => {
    const result = buildFieldTypeMap([
      [{ name: "a", type: "string" }],
      [{ name: "b", type: "array", array_type: "integer" }],
    ]);
    expect(result.fieldType.get("a")).toBe("string");
    expect(result.fieldType.get("b")).toBe("array");
    expect(result.arrayElementType.get("b")).toBe("integer");
    expect(result.arrayElementType.has("a")).toBe(false);
  });
});

// =============================================================================
// MILVUS — REGRESSION: behavior must match the pre-PR implementation exactly,
// since callers that hit a Milvus deployment must continue to receive the same
// filter_expr string they always have.
// =============================================================================

describe("compileMilvusFilter", () => {
  it("returns undefined for an empty list", () => {
    expect(compileMilvusFilter([], emptyTypes)).toBeUndefined();
  });

  it("emits Milvus syntax for a string equality clause", () => {
    const filters: Filter[] = [
      { field: "category", operator: "==", value: "AI" },
    ];
    expect(compileMilvusFilter(filters, emptyTypes)).toBe(
      'content_metadata["category"] == "AI"'
    );
  });

  it("does not quote numeric values", () => {
    const filters: Filter[] = [
      { field: "priority", operator: ">", value: 5 },
    ];
    expect(compileMilvusFilter(filters, emptyTypes)).toBe(
      'content_metadata["priority"] > 5'
    );
  });

  it("formats array values with proper element quoting", () => {
    const filters: Filter[] = [
      { field: "tags", operator: "in", value: ["ai", "ml"] },
    ];
    expect(compileMilvusFilter(filters, emptyTypes)).toBe(
      'content_metadata["tags"] in ["ai", "ml"]'
    );
  });

  it("leaves numeric array elements unquoted when the field is array<integer>", () => {
    const types = buildFieldTypeMap([
      [{ name: "scores", type: "array", array_type: "integer" }],
    ]);
    const filters: Filter[] = [
      { field: "scores", operator: "in", value: [1, 2, 3] },
    ];
    expect(compileMilvusFilter(filters, types)).toBe(
      'content_metadata["scores"] in [1, 2, 3]'
    );
  });

  it("uses function-call syntax for array_contains family operators", () => {
    const filters: Filter[] = [
      { field: "tags", operator: "array_contains", value: "engineering" },
    ];
    expect(compileMilvusFilter(filters, emptyTypes)).toBe(
      'array_contains(content_metadata["tags"], "engineering")'
    );
  });

  it("joins multiple clauses with the per-filter logical operator (default OR)", () => {
    const filters: Filter[] = [
      { field: "category", operator: "==", value: "AI" },
      {
        field: "priority",
        operator: ">",
        value: 5,
        logicalOperator: "AND",
      },
      { field: "category", operator: "==", value: "ML" },
    ];
    expect(compileMilvusFilter(filters, emptyTypes)).toBe(
      'content_metadata["category"] == "AI" and content_metadata["priority"] > 5 or content_metadata["category"] == "ML"'
    );
  });
});

// =============================================================================
// ELASTICSEARCH — new behavior. Must match the contract documented in
// docs/custom-metadata.md#elasticsearch-filter-example.
// =============================================================================

describe("compileElasticsearchFilter", () => {
  it("returns undefined for an empty list", () => {
    expect(compileElasticsearchFilter([])).toBeUndefined();
  });

  it("emits a `term` clause for string equality with the .keyword field path", () => {
    expect(
      compileElasticsearchFilter([
        { field: "category", operator: "==", value: "AI" },
      ])
    ).toEqual([
      { term: { "metadata.content_metadata.category.keyword": "AI" } },
    ]);
  });

  it("alias `=` produces the same `term` clause as `==`", () => {
    expect(
      compileElasticsearchFilter([
        { field: "category", operator: "=", value: "AI" },
      ])
    ).toEqual([
      { term: { "metadata.content_metadata.category.keyword": "AI" } },
    ]);
  });

  it("wraps inequality in bool.must_not", () => {
    expect(
      compileElasticsearchFilter([
        { field: "status", operator: "!=", value: "draft" },
      ])
    ).toEqual([
      {
        bool: {
          must_not: [
            { term: { "metadata.content_metadata.status.keyword": "draft" } },
          ],
        },
      },
    ]);
  });

  it.each([
    [">", "gt"],
    [">=", "gte"],
    ["<", "lt"],
    ["<=", "lte"],
  ] as const)(
    "maps numeric comparison %s to range.%s without .keyword (range never carries .keyword)",
    (operator, esKey) => {
      const result = compileElasticsearchFilter([
        { field: "priority", operator, value: 5 },
      ]);
      expect(result).toEqual([
        { range: { "metadata.content_metadata.priority": { [esKey]: 5 } } },
      ]);
    }
  );

  it("maps relative datetime operators to range with gt/lt, bare path", () => {
    expect(
      compileElasticsearchFilter([
        { field: "created", operator: "after", value: "2025-01-01" },
      ])
    ).toEqual([
      {
        range: {
          "metadata.content_metadata.created": { gt: "2025-01-01" },
        },
      },
    ]);
    expect(
      compileElasticsearchFilter([
        { field: "created", operator: "before", value: "2025-01-01" },
      ])
    ).toEqual([
      {
        range: {
          "metadata.content_metadata.created": { lt: "2025-01-01" },
        },
      },
    ]);
  });

  it("maps `in` to terms and `not in` to bool.must_not.terms", () => {
    expect(
      compileElasticsearchFilter([
        { field: "tags", operator: "in", value: ["ai", "ml"] },
      ])
    ).toEqual([
      {
        terms: {
          "metadata.content_metadata.tags.keyword": ["ai", "ml"],
        },
      },
    ]);
    expect(
      compileElasticsearchFilter([
        { field: "tags", operator: "not in", value: ["deprecated"] },
      ])
    ).toEqual([
      {
        bool: {
          must_not: [
            {
              terms: {
                "metadata.content_metadata.tags.keyword": ["deprecated"],
              },
            },
          ],
        },
      },
    ]);
  });

  it("translates Milvus `like` wildcards (% → *, _ → ?) to ES `wildcard`", () => {
    expect(
      compileElasticsearchFilter([
        { field: "title", operator: "like", value: "policy_%" },
      ])
    ).toEqual([
      {
        wildcard: {
          "metadata.content_metadata.title.keyword": "policy?*",
        },
      },
    ]);
  });

  it("emits term for array_contains and bool.must of terms for array_contains_all", () => {
    expect(
      compileElasticsearchFilter([
        { field: "tags", operator: "array_contains", value: "engineering" },
      ])
    ).toEqual([
      {
        term: {
          "metadata.content_metadata.tags.keyword": "engineering",
        },
      },
    ]);
    expect(
      compileElasticsearchFilter([
        {
          field: "tags",
          operator: "array_contains_all",
          value: ["tech", "ai"],
        },
      ])
    ).toEqual([
      {
        bool: {
          must: [
            { term: { "metadata.content_metadata.tags.keyword": "tech" } },
            { term: { "metadata.content_metadata.tags.keyword": "ai" } },
          ],
        },
      },
    ]);
  });

  it("emits terms for array_contains_any / includes; bool.must_not for `does not include`", () => {
    expect(
      compileElasticsearchFilter([
        {
          field: "tags",
          operator: "array_contains_any",
          value: ["tech", "ai"],
        },
      ])
    ).toEqual([
      {
        terms: { "metadata.content_metadata.tags.keyword": ["tech", "ai"] },
      },
    ]);
    expect(
      compileElasticsearchFilter([
        { field: "tags", operator: "includes", value: ["alpha"] },
      ])
    ).toEqual([
      { terms: { "metadata.content_metadata.tags.keyword": ["alpha"] } },
    ]);
    expect(
      compileElasticsearchFilter([
        {
          field: "tags",
          operator: "does not include",
          value: ["deprecated"],
        },
      ])
    ).toEqual([
      {
        bool: {
          must_not: [
            {
              terms: {
                "metadata.content_metadata.tags.keyword": ["deprecated"],
              },
            },
          ],
        },
      },
    ]);
  });

  // Grouping rules ----------------------------------------------------------

  it("produces a flat list when all filters are AND-joined (matches doc example)", () => {
    expect(
      compileElasticsearchFilter([
        { field: "category", operator: "==", value: "AI" },
        {
          field: "priority",
          operator: ">",
          value: 5,
          logicalOperator: "AND",
        },
      ])
    ).toEqual([
      { term: { "metadata.content_metadata.category.keyword": "AI" } },
      {
        range: {
          "metadata.content_metadata.priority": { gt: 5 },
        },
      },
    ]);
  });

  it("wraps once in bool.should when all subsequent joins are OR", () => {
    const result = compileElasticsearchFilter([
      { field: "category", operator: "==", value: "AI" },
      {
        field: "category",
        operator: "==",
        value: "ML",
        logicalOperator: "OR",
      },
    ]);
    expect(result).toEqual([
      {
        bool: {
          should: [
            { term: { "metadata.content_metadata.category.keyword": "AI" } },
            { term: { "metadata.content_metadata.category.keyword": "ML" } },
          ],
        },
      },
    ]);
  });

  it("groups AND-runs inside bool.should for mixed AND/OR (default Milvus precedence)", () => {
    // (a==1) OR (b==2 AND c==3) — the AND-run after the OR is one group.
    const result = compileElasticsearchFilter([
      { field: "a", operator: "==", value: 1 },
      { field: "b", operator: "==", value: 2, logicalOperator: "OR" },
      { field: "c", operator: "==", value: 3, logicalOperator: "AND" },
    ]);
    expect(result).toEqual([
      {
        bool: {
          should: [
            { term: { "metadata.content_metadata.a.keyword": 1 } },
            {
              bool: {
                must: [
                  { term: { "metadata.content_metadata.b.keyword": 2 } },
                  { term: { "metadata.content_metadata.c.keyword": 3 } },
                ],
              },
            },
          ],
        },
      },
    ]);
  });

  // --- Type-aware path selection ------------------------------------------
  // `.keyword` is an ES multi-field that only exists on string mappings.
  // Targeting `<numeric_field>.keyword` returns zero hits silently. The
  // compiler must consult the schema and omit `.keyword` for non-string
  // fields. Regression coverage for the case reported by the user where
  // an integer `priority` filter from the UI matched nothing.

  describe("type-aware field paths from schema", () => {
    it("omits .keyword for integer term equality (the reported bug)", () => {
      const types = buildFieldTypeMap([
        [{ name: "priority", type: "integer" }],
      ]);
      const result = compileElasticsearchFilter(
        [{ field: "priority", operator: "==", value: 2 }],
        types
      );
      expect(result).toEqual([
        { term: { "metadata.content_metadata.priority": 2 } },
      ]);
    });

    it("omits .keyword for float term equality", () => {
      const types = buildFieldTypeMap([
        [{ name: "rating", type: "float" }],
      ]);
      const result = compileElasticsearchFilter(
        [{ field: "rating", operator: "==", value: 4.5 }],
        types
      );
      expect(result).toEqual([
        { term: { "metadata.content_metadata.rating": 4.5 } },
      ]);
    });

    it("omits .keyword for boolean term equality", () => {
      const types = buildFieldTypeMap([
        [{ name: "is_public", type: "boolean" }],
      ]);
      const result = compileElasticsearchFilter(
        [{ field: "is_public", operator: "==", value: true }],
        types
      );
      expect(result).toEqual([
        { term: { "metadata.content_metadata.is_public": true } },
      ]);
    });

    it("omits .keyword for datetime range", () => {
      const types = buildFieldTypeMap([
        [{ name: "created_at", type: "datetime" }],
      ]);
      const result = compileElasticsearchFilter(
        [{ field: "created_at", operator: "after", value: "2025-01-01" }],
        types
      );
      expect(result).toEqual([
        {
          range: {
            "metadata.content_metadata.created_at": { gt: "2025-01-01" },
          },
        },
      ]);
    });

    it("omits .keyword for integer terms (in)", () => {
      const types = buildFieldTypeMap([
        [{ name: "year", type: "integer" }],
      ]);
      const result = compileElasticsearchFilter(
        [{ field: "year", operator: "in", value: [2024, 2025] }],
        types
      );
      expect(result).toEqual([
        {
          terms: {
            "metadata.content_metadata.year": [2024, 2025],
          },
        },
      ]);
    });

    it("keeps .keyword for string term equality when schema is known", () => {
      const types = buildFieldTypeMap([
        [{ name: "status", type: "string" }],
      ]);
      const result = compileElasticsearchFilter(
        [{ field: "status", operator: "==", value: "approved" }],
        types
      );
      expect(result).toEqual([
        { term: { "metadata.content_metadata.status.keyword": "approved" } },
      ]);
    });

    it("keeps .keyword for array<string> terms (in)", () => {
      const types = buildFieldTypeMap([
        [{ name: "tags", type: "array", array_type: "string" }],
      ]);
      const result = compileElasticsearchFilter(
        [{ field: "tags", operator: "in", value: ["ai", "ml"] }],
        types
      );
      expect(result).toEqual([
        {
          terms: { "metadata.content_metadata.tags.keyword": ["ai", "ml"] },
        },
      ]);
    });

    it("omits .keyword for array<integer> terms (in)", () => {
      const types = buildFieldTypeMap([
        [{ name: "scores", type: "array", array_type: "integer" }],
      ]);
      const result = compileElasticsearchFilter(
        [{ field: "scores", operator: "in", value: [1, 2, 3] }],
        types
      );
      expect(result).toEqual([
        { terms: { "metadata.content_metadata.scores": [1, 2, 3] } },
      ]);
    });

    it("defaults to string-like (keep .keyword) when the field is missing from the schema", () => {
      // Back-compat: the BE normalizer will strip .keyword server-side if
      // the field turns out to be non-string. This preserves the wire
      // format for installs that don't register metadata schemas.
      const types = buildFieldTypeMap([
        [{ name: "other_field", type: "string" }],
      ]);
      const result = compileElasticsearchFilter(
        [{ field: "unregistered", operator: "==", value: "x" }],
        types
      );
      expect(result).toEqual([
        { term: { "metadata.content_metadata.unregistered.keyword": "x" } },
      ]);
    });
  });
});

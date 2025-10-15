// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * Supported data types for metadata fields in collections.
 */
export type MetadataFieldType = 
  | "string" 
  | "integer" 
  | "float" 
  | "number" 
  | "boolean" 
  | "datetime" 
  | "array";

/**
 * Array element types for array metadata fields.
 */
export type ArrayElementType = "string" | "number" | "integer" | "float" | "boolean";

/**
 * API representation of metadata field (from backend).
 */
export interface APIMetadataField {
  name: string;
  type: MetadataFieldType;
  required?: boolean;
  array_type?: ArrayElementType; // Required for array types
  max_length?: number;
  description?: string;
}

/**
 * UI representation of metadata field (used in frontend forms).
 */
export interface UIMetadataField {
  name: string;
  type: MetadataFieldType;
  required?: boolean;
  array_type?: ArrayElementType; // Required for array types
  max_length?: number;
  description?: string;
  // Legacy support for existing code
  optional?: boolean; // Will be computed from !required
}

/**
 * Collection with enhanced metadata schema.
 */
export interface Collection {
  collection_name: string;
  num_entities: number;
  metadata_schema: APIMetadataField[];
}

/**
 * Filter structures for enhanced filtering.
 */
export interface FilterCondition {
  field: string;
  operator: FilterOperator;
  value: string | number | boolean | string[] | number[];
}

/**
 * Enhanced filter operators supporting all data types.
 */
export type FilterOperator = 
  // String operators
  | "==" | "=" | "!=" 
  | "like" | "LIKE"
  | "in" | "IN" | "not in" | "NOT IN"
  // Numeric operators  
  | ">" | ">=" | "<" | "<="
  | "between" | "BETWEEN"
  // Datetime operators
  | "before" | "BEFORE" | "after" | "AFTER"
  // Array operators
  | "array_contains" | "array_contains_all" | "array_contains_any"
  // Logical operators
  | "AND" | "OR" | "NOT";

// Natural language filter generation configuration
export interface FilterGenerationConfig {
  enable_filter_generator: boolean;
  model_name?: string;
  server_url?: string;
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
}

// Filter generation result
export interface GeneratedFilter {
  filter_expr: string;
  success: boolean;
  error?: string;
  reasoning?: string;
}
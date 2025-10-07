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
 * Represents a citation in a chat message response.
 */
export interface Citation {
  text: string;
  source: string;
  document_type: "text" | "image" | "table" | "chart";
  score?: number | string;
}

/**
 * Represents a message in the chat conversation.
 */
export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  citations?: Citation[];
  is_error?: boolean;
}

/**
 * Represents a filter for search queries.
 */
export interface Filter {
  field: string;
  operator: "=" | "!=" | ">" | "<" | ">=" | "<=" | "in" | "includes" | "does not include" | "like" | "not in" | "before" | "after" | "array_contains" | "array_contains_all" | "array_contains_any";
  value: string | number | boolean | (string | number | boolean)[];
  // Logical operator to join this filter with the previous one (undefined for first filter)
  logicalOperator?: "AND" | "OR";
}

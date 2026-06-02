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

import type { MessageContent } from "./chat";

/**
 * Request payload for the generate chat API endpoint.
 * Most fields are optional to avoid sending unnecessary defaults.
 * Only messages and use_knowledge_base are required.
 */
export interface GenerateRequest {
  // Required fields
  messages: { role: "user" | "assistant"; content: MessageContent }[];
  use_knowledge_base: boolean;
  
  // Optional RAG configuration
  collection_names?: string[];
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  reranker_top_k?: number;
  vdb_top_k?: number;
  confidence_threshold?: number;
  
  // Optional feature toggles
  enable_citations?: boolean;
  enable_guardrails?: boolean;
  enable_query_rewriting?: boolean;
  enable_reranker?: boolean;
  enable_vlm_inference?: boolean;
  enable_filter_generator?: boolean;
  
  // Optional models and endpoints
  model?: string;
  embedding_model?: string;
  reranker_model?: string;
  vlm_model?: string;
  llm_endpoint?: string;
  embedding_endpoint?: string;
  reranker_endpoint?: string;
  vlm_endpoint?: string;
  vdb_endpoint?: string;
  
  // Optional other fields.
  //
  // `filter_expr` shape depends on the configured vector store:
  //  - Milvus  → string expression (`content_metadata["x"] op v`).
  //  - Elasticsearch → list of dicts (Elasticsearch Query DSL). Field paths
  //    are `metadata.content_metadata.<name>`. A `.keyword` suffix is
  //    appended ONLY for exact-match clauses (term/terms/prefix/wildcard/
  //    match) on string-typed (or array<string>) fields — `.keyword` only
  //    exists as a multi-field on string mappings. Numeric, datetime,
  //    and boolean fields, plus all `range` clauses, use the bare path.
  // See docs/custom-metadata.md for the full contract.
  filter_expr?: string | Array<Record<string, unknown>>;
  stop?: string[];

  /**
   * Route this request through the agentic RAG pipeline.
   *
   * - `undefined` / omitted → server decides based on its own configuration
   *   (`CONFIG.enable_agentic_rag`).
   * - `true` → force the LangGraph plan-and-execute agentic pipeline.
   * - `false` → force the standard RAG pipeline.
   *
   * Mirrors `Prompt.agentic` on the server (`bool | None`).
   */
  agentic?: boolean | null;
}
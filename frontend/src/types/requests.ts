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
 * Request payload for the generate chat API endpoint.
 * Most fields are optional to avoid sending unnecessary defaults.
 * Only messages and use_knowledge_base are required.
 */
export interface GenerateRequest {
  // Required fields
  messages: { role: "user" | "assistant"; content: string }[];
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
  
  // Optional other fields
  filter_expr?: string;
  stop?: string[];
}
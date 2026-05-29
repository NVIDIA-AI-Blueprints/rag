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
  /**
   * Pipeline stage that produced this citation.
   *
   * Mirrors `SourceResult.stage` on the server. The server defaults to
   * `"rag"` for the standard pipeline; the agentic pipeline emits values
   * like `"initial_retrieval"`, `"execute"`, `"verify_execute"`, and may
   * add new values over time. Treated as an opaque string so we render
   * any future stage without code changes.
   */
  stage?: string;
}

/**
 * Text content for multimodal messages.
 */
export interface TextContent {
  type: "text";
  text: string;
}

/**
 * Image URL content for multimodal messages.
 */
export interface ImageContent {
  type: "image_url";
  image_url: {
    url: string;
    detail?: "auto" | "low" | "high";
  };
}

/**
 * Multimodal content can be text, image, or a combination.
 */
export type MessageContent = string | (TextContent | ImageContent)[];

/**
 * One entry in the streamed reasoning trace.
 *
 * Built incrementally by `useChatStream` from either agentic `event_type`
 * chunks or standard RAG chunks that carry `delta.reasoning_content`.
 * Standard responses without reasoning content leave `reasoning_steps`
 * undefined.
 *
 * The `stage` is treated as an opaque string (e.g. `"plan"`, `"execute"`,
 * `"synthesize"`, `"verify"`, `"verify_execute"`, `"initial_retrieval"`),
 * so any future graph node renders without code changes.
 */
export interface ReasoningStep {
  /**
   * Pipeline stage supplied by the server's `stage` field, or `"rag"` for
   * standard RAG reasoning.
   */
  stage: string;
  /** One-liner from the matching `stage_start` chunk. */
  label?: string;
  /** One-liner from the matching `stage_end` chunk. */
  summary?: string;
  /**
   * Concatenated `reasoning_content` from `intermediate_reasoning` and
   * `final_reasoning` chunks for this stage.
   */
  reasoning: string;
  /**
   * Concatenated `reasoning_content` from `intermediate_output` chunks
   * for this stage (planner / verifier JSON, per-task answers, ...).
   */
  output: string;
  /** Lifecycle: open between `stage_start` and `stage_end`. */
  status: "running" | "done" | "error";
}

/**
 * TTFT-style timings emitted on the trailing `finish_reason: "stop"` chunk.
 * All fields are optional so future metrics added by the server flow
 * through unchanged.
 */
export interface AgenticMetrics {
  rag_ttft_ms?: number;
  llm_ttft_ms?: number;
  llm_generation_time_ms?: number;
  [key: string]: number | undefined;
}

/**
 * Represents a message in the chat conversation.
 */
export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: MessageContent;
  timestamp: string;
  citations?: Citation[];
  is_error?: boolean;
  /**
   * Reasoning trace, populated when the server emits agentic `event_type`
   * chunks or standard RAG `delta.reasoning_content` chunks.
   */
  reasoning_steps?: ReasoningStep[];
  /** Trailing-chunk metrics from agentic streaming. */
  metrics?: AgenticMetrics;
}

/**
 * Represents a filter for search queries.
 */
export interface Filter {
  field: string;
  operator: "==" | "=" | "!=" | ">" | "<" | ">=" | "<=" | "in" | "includes" | "does not include" | "like" | "not in" | "before" | "after" | "array_contains" | "array_contains_all" | "array_contains_any";
  value: string | number | boolean | (string | number | boolean)[];
  // Logical operator to join this filter with the previous one (undefined for first filter)
  logicalOperator?: "AND" | "OR";
}

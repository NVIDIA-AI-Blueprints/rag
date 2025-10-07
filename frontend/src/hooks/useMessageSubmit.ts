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

import { useCallback } from "react";
import { useChatStore } from "../store/useChatStore";
import { useSendMessage } from "../api/useSendMessage";
import { useSettingsStore, useHealthDependentFeatures } from "../store/useSettingsStore";
import { useCollectionsStore } from "../store/useCollectionsStore";
import { useStreamingStore } from "../store/useStreamingStore";
import { useCollections } from "../api/useCollectionsApi";
import { useUUID } from "./useUUID";
import type { GenerateRequest } from "../types/requests";
import type { ChatMessage, Filter } from "../types/chat";
import type { Collection } from "../types/collections";

/**
 * Utility function to remove undefined, null, empty string, and empty array values from a request object.
 * This ensures we only send meaningful parameters to the API.
 */
function cleanRequestObject(obj: Partial<GenerateRequest>): GenerateRequest {
  const cleaned: Partial<GenerateRequest> = {};
  
  for (const [key, value] of Object.entries(obj)) {
    // Skip undefined, null, empty strings, and empty arrays
    if (value === undefined || value === null || value === "" || 
        (Array.isArray(value) && value.length === 0)) {
      continue;
    }
    
    // For boolean values, only include true values (skip false to avoid sending defaults)
    // Exception: always include use_knowledge_base since it's required
    if (typeof value === "boolean") {
      const alwaysInclude = ['use_knowledge_base'];
      if (value === true || alwaysInclude.includes(key)) {
        (cleaned as any)[key] = value;
      }
      // Skip false values for other boolean fields to avoid sending defaults
      continue;
    }
    
    (cleaned as any)[key] = value;
  }
  
  return cleaned as GenerateRequest;
}

/**
 * Custom hook for handling message submission in the chat interface.
 * 
 * Manages the complete message submission flow including validation,
 * message creation, settings integration, and API communication.
 * Handles user input, selected collections, filters, and streaming responses.
 * 
 * @returns Object with submit function and submission state
 * 
 * @example
 * ```tsx
 * const { handleSubmit } = useMessageSubmit();
 * handleSubmit(); // Submits current input as message
 * ```
 */
export const useMessageSubmit = () => {
  const { input, setInput, filters, addMessage, messages } = useChatStore();
  const { mutateAsync: sendMessage, resetStream } = useSendMessage();
  const { isStreaming } = useStreamingStore(); // Use centralized streaming state
  const { selectedCollections } = useCollectionsStore();
  const { data: allCollections = [] } = useCollections();
  const settings = useSettingsStore();
  const { generateUUID } = useUUID();
  const { shouldDisableHealthFeatures, isHealthLoading } = useHealthDependentFeatures();

  const createRequest = useCallback((currentMessages: ChatMessage[]) => {
    const rawRequest = {
      messages: currentMessages.map(({ role, content }) => ({ role, content })),
      use_knowledge_base: selectedCollections.length > 0,
      temperature: settings.temperature,
      top_p: settings.topP,
      max_tokens: settings.maxTokens,
      reranker_top_k: settings.rerankerTopK,
      vdb_top_k: settings.vdbTopK,
      vdb_endpoint: settings.vdbEndpoint,
      collection_names: selectedCollections.length > 0 ? selectedCollections : undefined,
      enable_query_rewriting: settings.enableQueryRewriting,
      enable_reranker: settings.enableReranker,
      enable_guardrails: settings.useGuardrails,
      enable_citations: settings.includeCitations,
      enable_vlm_inference: settings.enableVlmInference,
      enable_filter_generator: settings.enableFilterGenerator,
      model: settings.model,
      llm_endpoint: settings.llmEndpoint,
      embedding_model: settings.embeddingModel,
      embedding_endpoint: settings.embeddingEndpoint,
      reranker_model: settings.rerankerModel,
      reranker_endpoint: settings.rerankerEndpoint,
      vlm_model: settings.vlmModel,
      vlm_endpoint: settings.vlmEndpoint,
      stop: settings.stopTokens,
      confidence_threshold: settings.confidenceScoreThreshold,
      filter_expr: filters.length
        ? filters
            .map((f: Filter, index: number) => {
              // Create a map of field names to their types and array_types from selected collections
              const fieldTypeMap = new Map<string, string>();
              const fieldArrayTypeMap = new Map<string, string>();
              selectedCollections.forEach(collectionName => {
                const collection = allCollections.find((col: Collection) => col.collection_name === collectionName);
                if (collection?.metadata_schema) {
                  collection.metadata_schema.forEach((field: { name: string; type: string; description: string; array_type?: string }) => {
                    fieldTypeMap.set(field.name, field.type);
                    if (field.array_type) {
                      fieldArrayTypeMap.set(field.name, field.array_type);
                    }
                  });
                }
              });
              
              const isArrayField = fieldTypeMap.get(f.field) === 'array';
              const arrayElementType = fieldArrayTypeMap.get(f.field);
              
              const formatValue = (value: string | number | boolean | (string | number | boolean)[], isArrayField = false): string => {
                if (Array.isArray(value)) {
                  // Handle array values for operators like "in", "not in", "=", "!="
                  const formattedItems = value.map(item => {
                    if (typeof item === 'boolean' || typeof item === 'number') {
                      return String(item);
                    }
                    // For array fields, check the array_type to determine if we should quote string values
                    if (isArrayField && arrayElementType) {
                      // Don't quote if array elements are numeric or boolean types
                      if (['integer', 'float', 'number', 'boolean'].includes(arrayElementType)) {
                        return String(item);
                      }
                    }
                    // Quote string values (default behavior)
                    return `"${item}"`;
                  }).join(', ');
                  return `[${formattedItems}]`;
                }
                if (typeof value === 'boolean' || typeof value === 'number') {
                  return String(value); // true/false/numbers without quotes
                }
                return `"${value}"`; // strings with quotes
              };

              let filterExpression = '';
              // Handle special operators
              switch (f.operator) {
                case 'array_contains':
                case 'array_contains_all':
                case 'array_contains_any':
                  // Use function call syntax: array_contains(field, value)
                  filterExpression = `${f.operator}(content_metadata["${f.field}"], ${formatValue(f.value, isArrayField)})`;
                  break;
                default:
                  filterExpression = `content_metadata["${f.field}"] ${f.operator} ${formatValue(f.value, isArrayField)}`;
              }
              
              // Add logical operator prefix for all filters except the first one
              if (index > 0) {
                const logicalOp = f.logicalOperator || 'OR'; // Default to OR if not specified
                return ` ${logicalOp.toLowerCase()} ${filterExpression}`;
              }
              
              return filterExpression;
            })
            .join('')
        : undefined
    };
    
    // Clean the request object to remove undefined/empty values
    return cleanRequestObject(rawRequest);
  }, [selectedCollections, allCollections, settings, filters]);

  const handleSubmit = useCallback(async () => {
    if (!input.trim() || shouldDisableHealthFeatures || isStreaming) return;

    const userMessage = {
      id: generateUUID(),
      role: "user" as const,
      content: input,
      timestamp: new Date().toISOString(),
    };

    const assistantMessage = {
      id: generateUUID(),
      role: "assistant" as const,
      content: "",
      timestamp: new Date().toISOString(),
    };

    const currentMessages = [...messages, userMessage];
    addMessage(userMessage);
    addMessage(assistantMessage);
    setInput("");
    resetStream();

    const request = createRequest(currentMessages);
    await sendMessage({ request, assistantId: assistantMessage.id });
  }, [input, messages, addMessage, setInput, resetStream, createRequest, sendMessage, generateUUID, shouldDisableHealthFeatures, isStreaming]);

  return {
    handleSubmit,
    canSubmit: input.trim().length > 0 && !shouldDisableHealthFeatures && !isStreaming,
    isHealthLoading,
    shouldDisableHealthFeatures,
  };
}; 
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
import { useImageAttachmentStore } from "../store/useImageAttachmentStore";
import { useCollections } from "../api/useCollectionsApi";
import { useHealthStatus } from "../api/useHealthApi";
import { useUUID } from "./useUUID";
import type { GenerateRequest } from "../types/requests";
import type { ChatMessage, Filter, MessageContent, TextContent, ImageContent } from "../types/chat";
import type { Collection } from "../types/collections";
import {
  buildFieldTypeMap,
  compileElasticsearchFilter,
  compileMilvusFilter,
  vectorStoreFromHealthService,
} from "../utils/filterExpression";

/**
 * Build the per-request `filter_expr` value in the wire format the
 * configured backend expects (Milvus string vs. Elasticsearch list-of-dicts).
 *
 * The deployment is mono-store: we pick the backend from the first
 * `/health.databases[].service` entry. If health hasn't loaded yet (or the
 * label is unrecognized), we default to Milvus to preserve pre-existing
 * behavior.
 *
 * Exported for direct unit testing of the wire-format contract.
 */
export function buildFilterExpression(
  filters: Filter[],
  selectedCollections: string[],
  allCollections: Collection[],
  healthServiceLabel: string | undefined
): GenerateRequest["filter_expr"] {
  if (!filters.length) return undefined;

  const schemas = selectedCollections
    .map(
      (name) =>
        allCollections.find((c) => c.collection_name === name)
          ?.metadata_schema ?? []
    )
    .map((schema) =>
      schema.map((field) => ({
        name: field.name,
        type: field.type as string,
        array_type: field.array_type ?? undefined,
      }))
    );
  const fieldTypes = buildFieldTypeMap(schemas);

  const backend = vectorStoreFromHealthService(healthServiceLabel) ?? "milvus";
  if (backend === "elasticsearch") {
    // Pass `fieldTypes` so non-string-typed fields (integer / float /
    // datetime / boolean) get the bare field path. Appending `.keyword`
    // to those targets a non-existent ES sub-field and returns zero hits.
    return compileElasticsearchFilter(filters, fieldTypes);
  }
  return compileMilvusFilter(filters, fieldTypes);
}

/**
 * Utility function to remove undefined, null, empty string, and empty array values from a request object.
 * This ensures we only send meaningful parameters to the API.
 *
 * Exported so the wire-format guarantees (notably: `agentic: false` must be
 * preserved to override the server default, while `agentic: undefined` must
 * be omitted) can be unit-tested in isolation.
 */
export function cleanRequestObject(obj: Partial<GenerateRequest>): GenerateRequest {
  const cleaned: Partial<GenerateRequest> = {};
  
  for (const [key, value] of Object.entries(obj)) {
    // Skip undefined, null, empty strings, and empty arrays
    if (value === undefined || value === null || value === "" || 
        (Array.isArray(value) && value.length === 0)) {
      continue;
    }
    
    // For boolean values, always include feature toggles that users can explicitly set
    // These must be sent even when false to override backend defaults
    if (typeof value === "boolean") {
      const alwaysInclude = [
        'use_knowledge_base',
        'enable_reranker',
        'enable_citations',
        'enable_query_rewriting',
        'enable_guardrails',
        'enable_vlm_inference',
        'enable_filter_generator',
        // `agentic: false` must reach the server to force the standard
        // pipeline; otherwise it would be dropped and CONFIG.enable_agentic_rag
        // (which may be true) would silently take over.
        'agentic',
      ];
      if (value === true || alwaysInclude.includes(key)) {
        (cleaned as Record<string, unknown>)[key] = value;
      }
      continue;
    }
    
    (cleaned as Record<string, unknown>)[key] = value;
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
  const { attachedImages, clearAllImages } = useImageAttachmentStore();
  const { data: allCollections = [] } = useCollections();
  // The deployment is mono-store (Milvus or Elasticsearch). We pull the
  // backend label from /health.databases[0].service and translate that
  // into the wire format for filter_expr below.
  const { data: health } = useHealthStatus();
  const settings = useSettingsStore();
  const { generateUUID } = useUUID();
  const { shouldDisableHealthFeatures, isHealthLoading } = useHealthDependentFeatures();

  const createRequest = useCallback((currentMessages: ChatMessage[]) => {
    // Map the per-request agentic mode to the wire format:
    //   "on"  -> true   (force agentic LangGraph pipeline)
    //   "off" -> false  (force standard RAG pipeline)
    // The previous "auto" mode (omit the field, let the server decide) was
    // dropped per the #514 review — the FE now always pins the choice.
    // `agentic: false` must still survive `cleanRequestObject` (see the
    // `alwaysInclude` list there) so the user's "Standard" choice isn't
    // silently overridden by the server's `CONFIG.enable_agentic_rag`.
    const agentic = settings.agenticMode === "on";

    const rawRequest = {
      messages: currentMessages.map(({ role, content }) => ({ 
        role, 
        content: content as MessageContent 
      })),
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
      agentic,
      filter_expr: buildFilterExpression(
        filters,
        selectedCollections,
        allCollections,
        health?.databases?.[0]?.service
      ),
    };
    
    // Clean the request object to remove undefined/empty values
    return cleanRequestObject(rawRequest);
  }, [selectedCollections, allCollections, settings, filters, health?.databases]);

  const handleSubmit = useCallback(async () => {
    // Allow submit if there's text OR attached images
    const hasText = input.trim().length > 0;
    const hasImages = attachedImages.length > 0;
    
    if ((!hasText && !hasImages) || shouldDisableHealthFeatures || isStreaming) return;

    // Build multimodal content if images are attached
    // Trim input to remove any trailing whitespace/newlines that would cause blank lines in the chat bubble
    let content: MessageContent;
    const trimmedInput = input.trim();
    if (hasImages) {
      const contentParts: (TextContent | ImageContent)[] = [];
      
      // Add text part if there's text
      if (hasText) {
        contentParts.push({ type: "text" as const, text: trimmedInput });
      }
      
      // Add all image parts
      for (const image of attachedImages) {
        contentParts.push({
          type: "image_url" as const,
          image_url: { url: image.dataUri, detail: "auto" as const },
        });
      }
      
      content = contentParts;
    } else {
      content = trimmedInput;
    }

    const userMessage: ChatMessage = {
      id: generateUUID(),
      role: "user" as const,
      content,
      timestamp: new Date().toISOString(),
    };

    const assistantMessage: ChatMessage = {
      id: generateUUID(),
      role: "assistant" as const,
      content: "",
      timestamp: new Date().toISOString(),
    };

    const currentMessages = [...messages, userMessage];
    addMessage(userMessage);
    addMessage(assistantMessage);
    setInput("");
    clearAllImages(); // Clear all attached images after sending
    resetStream();

    const request = createRequest(currentMessages);
    await sendMessage({ request, assistantId: assistantMessage.id });
  }, [input, attachedImages, messages, addMessage, setInput, clearAllImages, resetStream, createRequest, sendMessage, generateUUID, shouldDisableHealthFeatures, isStreaming]);

  return {
    handleSubmit,
    canSubmit: (input.trim().length > 0 || attachedImages.length > 0) && !shouldDisableHealthFeatures && !isStreaming,
    isHealthLoading,
    shouldDisableHealthFeatures,
  };
}; 
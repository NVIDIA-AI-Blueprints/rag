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

import { useState, useRef } from "react";
import type { ChatMessage, Citation } from "../types/chat";

/**
 * Interface representing the state of a chat stream.
 */
export interface StreamState {
  content: string;
  citations: ChatMessage["citations"];
  error: string | null;
  isTyping: boolean;
}

/**
 * Custom hook for managing chat streaming functionality.
 * 
 * Provides utilities to start, process, and stop streaming chat responses.
 * Handles streaming text content, citations, and error states.
 * 
 * @returns An object containing stream state and control functions
 * 
 * @example
 * ```tsx
 * const { streamState, startStream, processStream, stopStream } = useChatStream();
 * const controller = startStream();
 * await processStream(response, messageId, updateMessage, threshold);
 * ```
 */
export const useChatStream = () => {
  const [streamState, setStreamState] = useState<StreamState>({
    content: "",
    citations: [],
    error: null,
    isTyping: false,
  });

  const abortControllerRef = useRef<AbortController | null>(null);

  const startStream = () => {
    const controller = new AbortController();
    abortControllerRef.current = controller;
    setStreamState({ content: "", citations: [], error: null, isTyping: true });
    return controller;
  };

  const stopStream = () => {
    abortControllerRef.current?.abort();
    setStreamState((prev) => ({ ...prev, isTyping: false }));
  };

  const resetStream = () => {
    setStreamState({ content: "", citations: [], error: null, isTyping: false });
  };

  const processStream = async (
    response: Response,
    assistantId: string,
    updateMessage: (id: string, update: Partial<ChatMessage>) => void
  ) => {
    const reader = response.body?.getReader();
    if (!reader) throw new Error("No response body in stream");

    const decoder = new TextDecoder();
    let buffer = "";
    let content = "";
    let latestCitations: ChatMessage["citations"] = [];
    
    // Detect if this is an error response based on HTTP status code
    const isError = response.status >= 400;

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;

          const json = JSON.parse(line.slice(6));
          const delta = json.choices?.[0]?.delta?.content;
          if (delta) content += delta;

          // Extract citations from any expected location
          const sources =
            json.citations?.results ??
            json.sources?.results ??
            json.choices?.[0]?.message?.citations ??
            json.choices?.[0]?.message?.sources ??
            [];

          if (Array.isArray(sources)) {
            const scored: ChatMessage["citations"] = sources.map((src: Record<string, unknown>) => {
              const score = typeof src.score === 'string' || typeof src.score === 'number' ? src.score :
                           typeof src.confidence_score === 'string' || typeof src.confidence_score === 'number' ? src.confidence_score :
                           typeof src.similarity_score === 'string' || typeof src.similarity_score === 'number' ? src.similarity_score :
                           undefined;
              return {
                text: String(src.content || src.text || ""),
                source: String(src.document_name || src.source || src.title || "Unknown"),
                document_type: (src.document_type as Citation["document_type"]) || "text",
                score,
              };
            });

            // Backend already filters by confidence threshold, so no need to filter here
            latestCitations = scored;
          }

          updateMessage(assistantId, {
            content,
            citations: latestCitations && latestCitations.length ? latestCitations : undefined,
            is_error: isError,
          });

          if (json.choices?.[0]?.finish_reason === "stop") {
            setStreamState({ content, citations: latestCitations, error: null, isTyping: false });
            return;
          }
        }
      }
    } catch (err: unknown) {
      if (err instanceof Error && err.name === "AbortError") return;
      setStreamState((prev) => ({
        ...prev,
        error: "Error processing stream",
        isTyping: false,
      }));
      throw err;
    } finally {
      reader.releaseLock();
    }
  };

  return {
    streamState,
    startStream,
    stopStream,
    resetStream,
    processStream,
    isStreaming: streamState.isTyping,
  };
};

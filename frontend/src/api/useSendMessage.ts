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

import { useMutation } from "@tanstack/react-query";
import { useChatStore } from "../store/useChatStore";
import { useChatStream } from "../hooks/useChatStream";
import { useStreamingStore } from "../store/useStreamingStore";
import type { GenerateRequest } from "../types/requests";

/**
 * Arguments for sending a message.
 */
interface SendMessageArgs {
  request: GenerateRequest;
  assistantId: string;
}

/**
 * Custom hook for sending messages to the chat API with streaming support.
 * 
 * Handles the complete flow of sending a message, processing the streaming response,
 * and updating the chat state. Manages streaming indicators and error handling.
 * 
 * @returns A React Query mutation object for sending messages
 * 
 * @example
 * ```tsx
 * const { mutate: sendMessage, isPending } = useSendMessage();
 * sendMessage({ 
 *   request: { query: "Hello", collection_names: ["docs"] },
 *   assistantId: "assistant-1"
 * });
 * ```
 */
export const useSendMessage = () => {
  const { updateMessage } = useChatStore();
  const { processStream, startStream, stopStream, resetStream } = useChatStream();
  const { setStreaming, clearStreaming } = useStreamingStore();

  const mutation = useMutation<void, Error, SendMessageArgs>({
    mutationFn: async ({ request, assistantId }) => {
      resetStream();
      setStreaming(true, assistantId);
      const controller = startStream();

      const res = await fetch(`/api/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
        signal: controller.signal,
      });

      await processStream(
        res,
        assistantId,
        updateMessage
      );
      
      clearStreaming();
    },
    onError: (err, vars) => {
      clearStreaming();
      
      // Extract error message from the error object
      let errorMessage = "Sorry, there was an error processing your request.";
      
      if (err instanceof Error) {
        // Use the actual error message from the backend
        errorMessage = err.message;
      } else if (typeof err === 'string') {
        errorMessage = err;
      }
      
      updateMessage(vars.assistantId, {
        content: errorMessage,
      });
    },
  });

  return {
    ...mutation,
    stopStream,
    resetStream,
    isStreaming: mutation.isPending,
  };
};

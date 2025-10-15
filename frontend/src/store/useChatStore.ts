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

import { create } from "zustand";
import type { ChatMessage, Filter } from "../types/chat";

/**
 * State interface for the chat store.
 */
interface ChatState {
  messages: ChatMessage[];
  input: string;
  filters: Filter[];
  setInput: (value: string) => void;
  setFilters: (filters: Filter[]) => void;
  addMessage: (msg: ChatMessage) => void;
  updateMessage: (id: string, update: Partial<ChatMessage>) => void;
  clearMessages: () => void;
  reset: () => void;
}

/**
 * Zustand store for managing chat state including messages, input, and filters.
 * 
 * @returns Chat store with messages, input state, and actions
 * 
 * @example
 * ```tsx
 * const { messages, addMessage, setInput } = useChatStore();
 * addMessage({ id: '1', content: 'Hello', role: 'user' });
 * ```
 */
export const useChatStore = create<ChatState>((set) => ({
  messages: [],
  input: "",
  filters: [],
  setInput: (input) => set({ input }),
  setFilters: (filters) => set({ filters }),
  addMessage: (msg) => set((s) => ({ messages: [...s.messages, msg] })),
  updateMessage: (id, update) =>
    set((s) => ({
      messages: s.messages.map((m) =>
        m.id === id
          ? {
              ...m,
              ...update,
              citations:
                update.citations !== undefined ? update.citations : m.citations,
            }
          : m
      ),
    })),
  clearMessages: () => set({ messages: [] }),
  reset: () => set({ messages: [], input: "", filters: [] }),
}));

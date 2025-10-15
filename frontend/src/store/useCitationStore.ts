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

/**
 * State interface for the citation store.
 */
interface CitationState {
  expandedItems: Set<string>;
  toggleExpanded: (citationId: string) => void;
  clearExpanded: () => void;
}

/**
 * Zustand store for managing citation expansion state.
 * 
 * Tracks which citations are expanded/collapsed in the UI,
 * providing toggle and clear functionality for citation displays.
 * 
 * @returns Store with citation expansion state and actions
 */
export const useCitationStore = create<CitationState>((set) => ({
  expandedItems: new Set(),
  toggleExpanded: (citationId) => 
    set((state) => {
      const newExpanded = new Set(state.expandedItems);
      if (newExpanded.has(citationId)) {
        newExpanded.delete(citationId);
      } else {
        newExpanded.add(citationId);
      }
      return { expandedItems: newExpanded };
    }),
  clearExpanded: () => set({ expandedItems: new Set() }),
})); 
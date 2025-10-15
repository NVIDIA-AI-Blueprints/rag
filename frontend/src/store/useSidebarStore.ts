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
import type { Citation } from "../types/chat";

/**
 * Type definition for sidebar view states.
 */
type SidebarView = "citations" | null;

/**
 * State interface for the sidebar store.
 */
interface SidebarState {
  view: SidebarView;
  citations: Citation[];
  setActiveCitations: (c: Citation[]) => void;
  toggleSidebar: (view: SidebarView) => void;
  closeSidebar: () => void;
}

/**
 * Zustand store for managing sidebar state and content.
 * 
 * Controls sidebar visibility, active view type, and content data such as
 * citations. Provides functions to toggle views, set content, and close sidebar.
 * 
 * @returns Store with sidebar state and control functions
 * 
 * @example
 * ```tsx
 * const { view, citations, setActiveCitations, toggleSidebar } = useSidebarStore();
 * setActiveCitations(citations);
 * toggleSidebar("citations");
 * ```
 */
export const useSidebarStore = create<SidebarState>((set) => ({
  view: null,
  citations: [],
  setActiveCitations: (c) => set({ citations: c }),
  toggleSidebar: (view) => set({ view }),
  closeSidebar: () => set({ view: null, citations: [] }),
}));

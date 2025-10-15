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
import type { Collection } from "../types/collections";

/**
 * State interface for the collection drawer store.
 */
interface CollectionDrawerState {
  isOpen: boolean;
  activeCollection: Collection | null;
  showUploader: boolean;
  deleteError: string | null;
  
  openDrawer: (collection: Collection) => void;
  closeDrawer: () => void;
  toggleUploader: (show?: boolean) => void;
  setDeleteError: (error: string | null) => void;
  reset: () => void;
}

/**
 * Zustand store for managing collection drawer state and interactions.
 * 
 * Manages the sidebar drawer that displays collection details, document upload
 * functionality, and error states. Tracks which collection is active and
 * controls drawer visibility and uploader visibility.
 * 
 * @returns Store with drawer state and action functions
 * 
 * @example
 * ```tsx
 * const { isOpen, activeCollection, openDrawer, closeDrawer } = useCollectionDrawerStore();
 * openDrawer(collection);
 * ```
 */
export const useCollectionDrawerStore = create<CollectionDrawerState>((set) => ({
  isOpen: false,
  activeCollection: null,
  showUploader: false,
  deleteError: null,

  openDrawer: (collection) => 
    set({ isOpen: true, activeCollection: collection, showUploader: false }),
  
  closeDrawer: () => 
    set({ isOpen: false, activeCollection: null, showUploader: false, deleteError: null }),
  
  toggleUploader: (show) => 
    set((state) => ({ showUploader: show ?? !state.showUploader })),
  
  setDeleteError: (error) => 
    set({ deleteError: error }),
  
  reset: () => 
    set({ isOpen: false, activeCollection: null, showUploader: false, deleteError: null }),
})); 
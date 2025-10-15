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

// useCollectionsStore.ts
import { create } from "zustand";
import type { MetadataFieldType } from "../types/collections";

/**
 * Collection interface for the collections store.
 */
interface Collection {
  collection_name: string;
  num_entities: number;
  metadata_schema: { name: string; type: string; description: string }[];
}

/**
 * Valid Badge colors from KUI
 */
export type BadgeColor = "gray" | "blue" | "green" | "red" | "yellow" | "purple" | "teal";

/**
 * Color mapping for different field types
 */
export const FIELD_TYPE_COLORS: Record<MetadataFieldType, BadgeColor> = {
  string: "blue",
  integer: "yellow", 
  float: "yellow",
  number: "yellow",
  boolean: "purple",
  datetime: "red",
  array: "teal"
};

/**
 * State interface for the collections store.
 */
interface CollectionsState {
  selectedCollections: string[];
  drawerCollection: Collection | null;
  setSelectedCollections: (collections: string[]) => void;
  toggleCollection: (id: string) => void;
  clearCollections: () => void;
  setDrawerCollection: (col: Collection | null) => void;
  refresh: () => void;
  getFieldTypeColor: (fieldType: MetadataFieldType) => BadgeColor;
}

/**
 * Zustand store for managing collection selection and drawer state.
 * 
 * @returns Store with collection selection state and actions
 */
export const useCollectionsStore = create<CollectionsState>((set) => ({
  selectedCollections: [],
  drawerCollection: null,
  setSelectedCollections: (collections) => set({ selectedCollections: collections }),
  toggleCollection: (id) =>
    set((state) => {
      const exists = state.selectedCollections.includes(id);
      return {
        selectedCollections: exists
          ? state.selectedCollections.filter((c) => c !== id)
          : [...state.selectedCollections, id],
      };
    }),
  clearCollections: () => set({ selectedCollections: [] }),
  setDrawerCollection: (col) => set({ drawerCollection: col }),
  refresh: () => set({}),
  getFieldTypeColor: (fieldType) => FIELD_TYPE_COLORS[fieldType] || "gray",
}));

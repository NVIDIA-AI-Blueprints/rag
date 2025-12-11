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
import { persist } from "zustand/middleware";

/**
 * Per-collection configuration settings.
 */
export interface CollectionConfig {
  /** Whether to generate summaries when uploading documents to this collection */
  generateSummary: boolean;
}

/**
 * Default configuration for new collections.
 */
const defaultCollectionConfig: CollectionConfig = {
  generateSummary: true,
};

/**
 * State interface for collection configuration management.
 */
interface CollectionConfigState {
  /** Map of collection name to its configuration */
  configs: Record<string, CollectionConfig>;
  
  /** Get configuration for a specific collection (returns defaults if not set) */
  getConfig: (collectionName: string) => CollectionConfig;
  
  /** Update configuration for a specific collection */
  setConfig: (collectionName: string, config: Partial<CollectionConfig>) => void;
  
  /** Remove configuration for a collection (e.g., when collection is deleted) */
  removeConfig: (collectionName: string) => void;
}

/**
 * Zustand store for managing per-collection configuration settings.
 * 
 * Persists settings to localStorage so collection preferences are remembered
 * across sessions.
 * 
 * @example
 * ```tsx
 * const { getConfig, setConfig } = useCollectionConfigStore();
 * 
 * // Get config (returns defaults if not set)
 * const config = getConfig("my-collection");
 * console.log(config.generateSummary); // true by default
 * 
 * // Update config
 * setConfig("my-collection", { generateSummary: false });
 * ```
 */
export const useCollectionConfigStore = create<CollectionConfigState>()(
  persist(
    (set, get) => ({
      configs: {},
      
      getConfig: (collectionName: string) => {
        const { configs } = get();
        return configs[collectionName] ?? { ...defaultCollectionConfig };
      },
      
      setConfig: (collectionName: string, config: Partial<CollectionConfig>) => {
        set((state) => ({
          configs: {
            ...state.configs,
            [collectionName]: {
              ...defaultCollectionConfig,
              ...state.configs[collectionName],
              ...config,
            },
          },
        }));
      },
      
      removeConfig: (collectionName: string) => {
        set((state) => {
          // eslint-disable-next-line @typescript-eslint/no-unused-vars
          const { [collectionName]: _removed, ...rest } = state.configs;
          return { configs: rest };
        });
      },
    }),
    {
      name: "rag-collection-configs",
    }
  )
);


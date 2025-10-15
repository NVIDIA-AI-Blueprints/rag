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

import { useQuery } from "@tanstack/react-query";
import type { DocumentItem } from "../types/api";

/**
 * Custom hook to fetch unique metadata values for a specific field across collections.
 * 
 * @param collectionNames - Array of collection names to search
 * @param field - The metadata field to get values for
 * @returns React Query object containing unique metadata values
 * 
 * @example
 * ```tsx
 * const { data: categories } = useMetadataValues(["docs", "papers"], "category");
 * ```
 */
export function useMetadataValues(collectionNames: string[], field: string) {
  return useQuery<string[]>({
    queryKey: ["metadata-values", collectionNames.sort().join("|"), field],
    enabled: collectionNames.length > 0 && !!field,
    staleTime: 5 * 60 * 1000,
    queryFn: async () => {
      const seen = new Set<string>();
      for (const name of collectionNames) {
        const res = await fetch(
          `/api/documents?collection_name=${encodeURIComponent(
            name
          )}`
        );
        if (!res.ok) throw new Error("Failed to fetch documents");
        const json = await res.json();
        (json.documents || []).forEach((doc: DocumentItem) => {
          const v = doc.metadata?.[field];
          if (v != null && v !== "") {
            // Handle arrays by adding each element individually
            if (Array.isArray(v)) {
              v.forEach(item => {
                if (item != null && item !== "") {
                  seen.add(String(item));
                }
              });
            } else {
              seen.add(String(v));
            }
          }
        });
      }
      return Array.from(seen).sort();
    },
  });
} 
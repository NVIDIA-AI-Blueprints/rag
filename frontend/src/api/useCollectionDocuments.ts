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

import { useMutation, useQuery } from "@tanstack/react-query";
import type { CollectionDocumentsResponse } from "../types/api";

/**
 * Custom hook to fetch documents from a specific collection.
 * 
 * @param collectionName - The name of the collection to fetch documents from
 * @returns A React Query object containing documents data, loading state, and error state
 * 
 * @example
 * ```tsx
 * const { data, isLoading, error } = useCollectionDocuments("my-collection");
 * ```
 */
export const useCollectionDocuments = (collectionName: string) =>
  useQuery<CollectionDocumentsResponse>({
    queryKey: ["collection-documents", collectionName],
    queryFn: async () => {
      const res = await fetch(
        `/api/documents?collection_name=${encodeURIComponent(collectionName)}`
      );
      if (!res.ok) throw new Error("Failed to fetch documents");
      return res.json();
    },
    enabled: !!collectionName,
  });

/**
 * Custom hook to delete all documents from a collection.
 * 
 * @returns A React Query mutation object for deleting all documents from a collection
 * 
 * @example
 * ```tsx
 * const { mutate: deleteAllDocs, isPending } = useDeleteAllDocuments();
 * deleteAllDocs("collection-name");
 * ```
 */
export function useDeleteAllDocuments() {
  return useMutation({
    mutationFn: async (collectionName: string) => {
      const res = await fetch(
        `/api/documents?collection_name=${encodeURIComponent(collectionName)}`,
        {
          method: "DELETE",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify([]),
        }
      );
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.message || "Failed to delete documents");
      }
      return res.json();
    },
  });
}

export function useDeleteDocument() {
  return useMutation({
    mutationFn: async ({ collectionName, documentName }: { collectionName: string; documentName: string }) => {
      const res = await fetch(
        `/api/documents?collection_name=${encodeURIComponent(collectionName)}`,
        {
          method: "DELETE",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify([documentName]),
        }
      );
      if (!res.ok) {
        let message = "Failed to delete document";
        try { const err = await res.json(); message = err.message || message; } catch { /* ignore parse errors */ }
        throw new Error(message);
      }
      return res.json();
    },
  });
}
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

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import type { CreateCollectionPayload } from "../types/api";

/**
 * Custom hook to fetch all available collections.
 *
 * @returns A React Query object containing the list of collections
 *
 * @example
 * ```tsx
 * const { data: collections, isLoading, error } = useCollections();
 * ```
 */
export function useCollections() {
  return useQuery({
    queryKey: ["collections"],
    queryFn: async () => {
      const res = await fetch("/api/collections");
      if (!res.ok) throw new Error("Failed to fetch collections");
      const json = await res.json();
      return json.collections;
    },
  });
}

/**
 * Custom hook to create a new collection.
 *
 * @returns A React Query mutation object for creating collections
 *
 * @example
 * ```tsx
 * const { mutate: createCollection, isPending } = useCreateCollection();
 * createCollection({
 *   collection_name: "my-collection",
 *   embedding_dimension: 1536,
 *   metadata_schema: [],
 *   vdb_endpoint: "http://localhost:8000"
 * });
 * ```
 */
export function useCreateCollection() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (payload: CreateCollectionPayload) => {
      const res = await fetch(`/api/collection`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error("Failed to create collection");
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["collections"] });
    },
  });
}

/**
 * Custom hook to delete a collection.
 *
 * @returns A React Query mutation object for deleting collections
 *
 * @example
 * ```tsx
 * const { mutate: deleteCollection, isPending } = useDeleteCollection();
 * deleteCollection("collection-name");
 * ```
 */
export function useDeleteCollection() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (collectionName: string) => {
      const res = await fetch(
        `/api/collections`,
        {
          method: "DELETE",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify([collectionName]),
        }
      );
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.message || "Failed to delete collection");
      }
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["collections"] });
    },
  });
}



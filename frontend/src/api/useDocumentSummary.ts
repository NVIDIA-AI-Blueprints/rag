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

export interface DocumentSummaryResponse {
  summary: string;
  file_name: string;
  collection_name: string;
  status: "SUCCESS" | "PENDING" | "IN_PROGRESS" | "FAILED" | "NOT_FOUND";
  message?: string;
  error?: string;
}

/**
 * Fetch document summary from the API.
 */
export function useDocumentSummary(collectionName: string, fileName: string) {
  return useQuery<DocumentSummaryResponse>({
    queryKey: ["document-summary", collectionName, fileName],
    queryFn: async () => {
      const params = new URLSearchParams();
      params.append("collection_name", collectionName);
      params.append("file_name", fileName);

      const res = await fetch(`/api/summary?${params.toString()}`);
      if (!res.ok) {
        // Return NOT_FOUND for 404, let other errors throw
        if (res.status === 404) {
          return { summary: "", file_name: fileName, collection_name: collectionName, status: "NOT_FOUND" as const };
        }
        throw new Error("Failed to fetch summary");
      }
      return res.json();
    },
    enabled: !!collectionName && !!fileName,
    staleTime: 5 * 60 * 1000, // Cache for 5 minutes
    retry: false, // Don't retry on failure
    // Poll every 10 seconds when summary is being generated
    refetchInterval: (query: { state: { data?: DocumentSummaryResponse } }) => {
      const data = query.state.data;
      // Poll if status is PENDING or IN_PROGRESS, stop polling otherwise
      if (!data || data.status === "PENDING" || data.status === "IN_PROGRESS") {
        return 10000; // 10 seconds
      }
      return false; // Stop polling for terminal states (SUCCESS, FAILED, NOT_FOUND)
    },
  });
}


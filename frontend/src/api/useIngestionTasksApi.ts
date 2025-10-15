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
import type { IngestionTask } from "../types/api";

/**
 * Custom hook to fetch and poll ingestion task status.
 * 
 * @param taskId - The ID of the ingestion task to monitor
 * @param enabled - Whether to enable the query and polling
 * @returns React Query object with task status and automatic polling
 * 
 * @example
 * ```tsx
 * const { data: task, isLoading } = useIngestionTasks("task-123", true);
 * ```
 */
export function useIngestionTasks(taskId: string, enabled: boolean) {
  return useQuery<IngestionTask>({
    queryKey: ["task-status", taskId],
    queryFn: async () => {
      const res = await fetch(
        `/api/status?task_id=${taskId}`
      );
      if (!res.ok) throw new Error("Failed to fetch task status");
      return res.json();
    },
    enabled,
    refetchInterval: 5000,
  });
}

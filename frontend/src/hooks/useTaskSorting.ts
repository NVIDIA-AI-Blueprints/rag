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

import { useMemo } from "react";
import type { IngestionTask } from "../types/api";

/**
 * Extended task type with read status for sorting purposes.
 */
type TaskWithReadStatus = IngestionTask & { completedAt?: number; read?: boolean };

/**
 * Custom hook for sorting ingestion tasks by priority and read status.
 * 
 * Sorts tasks to prioritize unread tasks at the top, followed by read tasks.
 * Within each group, tasks are sorted by creation date (newest first).
 * 
 * @param pendingTasks - Array of pending ingestion tasks
 * @param completedTasks - Array of completed tasks with metadata
 * @returns Object with sorted tasks array
 * 
 * @example
 * ```tsx
 * const { sortedTasks } = useTaskSorting(pending, completed);
 * ```
 */
export const useTaskSorting = (
  pendingTasks: IngestionTask[],
  completedTasks: { key: string; task: TaskWithReadStatus }[]
) => {
  const sortedTasks = useMemo(() => {
    const allTasks: TaskWithReadStatus[] = [
      ...pendingTasks,
      ...completedTasks.map(e => e.task),
    ];

    return allTasks.sort((a, b) => {
      // First priority: unread tasks go to top
      const aUnread = a.state === "PENDING" || (a as unknown as { read?: boolean }).read !== true;
      const bUnread = b.state === "PENDING" || (b as unknown as { read?: boolean }).read !== true;
      
      if (aUnread && !bUnread) return -1;
      if (!aUnread && bUnread) return 1;
      
      // Within same read status, sort by newest first
      return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
    });
  }, [pendingTasks, completedTasks]);

  return { sortedTasks };
}; 
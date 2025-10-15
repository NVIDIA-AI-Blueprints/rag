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

import { useEffect, useRef } from "react";
import { useNotificationStore } from "../../store/useNotificationStore";
import { useIngestionTasks } from "../../api/useIngestionTasksApi";

/**
 * Props for the TaskPoller component.
 */
interface TaskPollerProps {
  taskId: string;
}

/**
 * Component that polls for ingestion task status updates.
 * 
 * Automatically polls the task status API for a specific task and updates
 * the task store with the latest status. Handles task completion detection
 * and cleanup of polling when tasks are finished.
 * 
 * @param props - Component props with task ID to poll
 * @returns Task poller component (renders nothing visible)
 */
export const TaskPoller = ({ taskId }: TaskPollerProps) => {
  const { updateTaskNotification, getAllNotifications } = useNotificationStore();
  const { data, isLoading, error } = useIngestionTasks(taskId, true);
  const lastUpdateRef = useRef<string>("");

  useEffect(() => {
    if (!data || isLoading || error) return;

    // Get the existing task to preserve the collection name if API doesn't return it
    const allNotifications = getAllNotifications();
    const existingTaskNotification = allNotifications.find(n => n.type === "task" && n.task.id === taskId);
    const existingTask = existingTaskNotification?.type === "task" ? existingTaskNotification.task : undefined;
    
    const task = {
      ...data,
      id: taskId,
      collection_name: data.collection_name || existingTask?.collection_name || "Unknown Collection",
    };

    // Create a hash of the important task data to check if anything actually changed
    const taskHash = JSON.stringify({
      state: task.state,
      collection_name: task.collection_name,
      documents: task.documents,
      result: task.result
    });

    // Only update if the task data has actually changed
    if (taskHash !== lastUpdateRef.current) {
      lastUpdateRef.current = taskHash;
      
      // Update task with latest data from API
      if (task.state !== "PENDING") {
        // Task completed - completedAt is handled automatically by the store
        updateTaskNotification(taskId, task);
      } else {
        // Update pending task
        updateTaskNotification(taskId, task);
      }
    }
  }, [data, isLoading, error, taskId, updateTaskNotification, getAllNotifications]);

  return null;
}; 
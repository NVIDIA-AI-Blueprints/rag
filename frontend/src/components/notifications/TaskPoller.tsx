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
import { useQueryClient } from "@tanstack/react-query";
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
  const queryClient = useQueryClient();
  const { updateTaskNotification, getAllNotifications, removeNotification } = useNotificationStore();
  const { data, isLoading, error } = useIngestionTasks(taskId, true);
  const previousStateRef = useRef<string>("PENDING");
  const hasInitialized = useRef(false);
  const errorCountRef = useRef(0);

  // Handle errors - if task doesn't exist on backend, remove the orphaned notification
  useEffect(() => {
    if (error && !isLoading) {
      errorCountRef.current++;
      // After 3 consecutive errors, assume task doesn't exist and remove it
      // This handles orphaned tasks from previous deployments
      if (errorCountRef.current >= 3) {
        const taskNotificationId = `task-${taskId}`;
        removeNotification(taskNotificationId);
      }
    } else if (data) {
      // Reset error count on successful fetch
      errorCountRef.current = 0;
    }
  }, [error, isLoading, data, taskId, removeNotification]);

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

    // Only update on significant changes: state change or initial load
    // Progress changes (documents_completed incrementing) don't require notification updates
    // This prevents notification count flickering during document processing
    const hasStateChanged = task.state !== previousStateRef.current;
    const isInitialLoad = !hasInitialized.current;
    
    if (hasStateChanged || isInitialLoad) {
      hasInitialized.current = true;
      
      // Check if task just completed (state changed from PENDING to something else)
      const justCompleted = previousStateRef.current === "PENDING" && task.state !== "PENDING";
      previousStateRef.current = task.state;
      
      // Update task with latest data from API
      updateTaskNotification(taskId, task);
      
      // Invalidate collections query to refresh file counts after ingestion completes
      if (justCompleted) {
        queryClient.invalidateQueries({ queryKey: ["collections"] });
        if (task.collection_name) {
          queryClient.invalidateQueries({ queryKey: ["collection-documents", task.collection_name] });
        }
      }
    }
  }, [data, isLoading, error, taskId, updateTaskNotification, getAllNotifications, queryClient]);

  return null;
};

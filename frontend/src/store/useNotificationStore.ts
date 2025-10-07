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
import type { IngestionTask } from "../types/api";
import type { 
  AppNotification, 
  HealthNotification, 
  TaskNotification, 
  NotificationSeverity 
} from "../types/notifications";

/**
 * State interface for the unified notification store.
 */
interface NotificationState {
  // All notifications
  notifications: AppNotification[];
  
  // Hydration (only called once on app start)
  hydrate: () => void;
  
  // Health notification management
  addHealthNotification: (notification: Omit<HealthNotification, 'id' | 'type' | 'createdAt' | 'read' | 'dismissed'>) => void;
  clearServiceNotifications: (serviceName: string) => void;
  
  // Task notification management (for compatibility with existing system)
  addTaskNotification: (task: IngestionTask) => void;
  updateTaskNotification: (taskId: string, updates: Partial<IngestionTask>) => void;
  
  // General notification management
  markAsRead: (id: string) => void;
  dismissNotification: (id: string) => void;
  removeNotification: (id: string) => void;
  clearNotifications: () => void;
  cleanupDuplicates: () => void;
  
  // Getters for existing notification system compatibility
  getPendingTasks: () => (IngestionTask & { read?: boolean })[];
  getCompletedTasks: () => (IngestionTask & { completedAt?: number; read?: boolean })[];
  getHealthNotifications: () => HealthNotification[];
  getAllNotifications: () => AppNotification[];
  getUnreadCount: () => number;
}

/**
 * Generate a unique ID for notifications.
 * For task notifications, we use a deterministic ID based on the task ID to prevent duplicates.
 * For health notifications, we use a timestamp and random suffix for uniqueness.
 */
const generateNotificationId = (type: string, identifier: string): string => {
  if (type === "task") {
    // For task notifications, use a deterministic ID to prevent duplicates
    return `task-${identifier}`;
  }
  
  // For health and other notifications, use timestamp and random suffix for uniqueness
  const timestamp = Date.now();
  const randomSuffix = Math.random().toString(36).substring(2, 8);
  return `${type}-${identifier}-${timestamp}-${randomSuffix}`;
};

/**
 * Convert task to task notification.
 */
const taskToNotification = (task: IngestionTask): TaskNotification => {
  // Determine severity based on task state
  let severity: NotificationSeverity = "info";
  if (task.state === "FAILED") severity = "error";
  else if (task.state === "FINISHED") severity = "success";
  else if (task.state === "PENDING") severity = "info";

  return {
    id: generateNotificationId("task", task.id),
    type: "task",
    title: `Collection: ${task.collection_name}`,
    message: task.state === "FINISHED" 
      ? `Upload completed successfully`
      : task.state === "FAILED"
      ? `Upload failed`
      : `Uploading documents...`,
    severity,
    createdAt: new Date(task.created_at).getTime(),
    read: false,
    dismissed: false,
    task: { 
      ...task, 
      completedAt: task.state !== "PENDING" ? Date.now() : undefined 
    },
    collectionName: task.collection_name,
  };
};

/**
 * Extract task from task notification for compatibility.
 */
const notificationToTask = (notification: TaskNotification): IngestionTask & { completedAt?: number; read?: boolean } => {
  return {
    ...notification.task,
    read: notification.read,
  };
};

/**
 * Unified notification store that manages both health and task notifications.
 * 
 * Provides a single source of truth for all application notifications while
 * maintaining compatibility with the existing task notification system.
 * 
 * @returns Store with unified notification state and management functions
 */
export const useNotificationStore = create<NotificationState>((set, get) => ({
  notifications: [],

  hydrate: () => {
    const currentState = get();
    const notifications: AppNotification[] = [];
    const seenTaskIds = new Set<string>();
    const taskKeys = new Map<string, string>(); // taskId -> localStorage key
    
    // First pass: collect all task keys and detect duplicates
    Object.keys(localStorage).forEach(key => {
      if (key.startsWith('completedTask:') || key.startsWith('ingestion-task-')) {
        try {
          const task = JSON.parse(localStorage.getItem(key)!);
          if (task?.id) {
            const existingKey = taskKeys.get(task.id);
            if (existingKey) {
              // Duplicate detected - prefer completedTask over ingestion-task
              if (key.startsWith('completedTask:') && existingKey.startsWith('ingestion-task-')) {
                localStorage.removeItem(existingKey); // Remove the old pending task
                taskKeys.set(task.id, key);
              } else if (key.startsWith('ingestion-task-') && existingKey.startsWith('completedTask:')) {
                localStorage.removeItem(key); // Remove this duplicate pending task
                return;
              } else {
                // Same type of key - remove the duplicate
                localStorage.removeItem(key);
                return;
              }
            } else {
              taskKeys.set(task.id, key);
            }
          }
        } catch {
          localStorage.removeItem(key); // Clean up corrupted entries
        }
      }
    });
    
    // Second pass: load notifications from cleaned keys
    taskKeys.forEach((key, taskId) => {
      try {
        const task = JSON.parse(localStorage.getItem(key)!);
        const taskNotification = taskToNotification(task);
        taskNotification.read = task.read || false;
        notifications.push(taskNotification);
        seenTaskIds.add(taskId);
      } catch {
        localStorage.removeItem(key); // Clean up corrupted entries
      }
    });

    // Only update state if notifications have actually changed
    const currentNotificationIds = new Set(currentState.notifications.map(n => n.id));
    const newNotificationIds = new Set(notifications.map(n => n.id));
    const hasChanges = currentState.notifications.length !== notifications.length ||
      ![...currentNotificationIds].every(id => newNotificationIds.has(id));

    if (hasChanges) {
      set({ notifications });
    }
  },

  addHealthNotification: (notificationData) => set((state) => {
    const { serviceName, severity, title } = notificationData;
    
    // Check if a similar notification already exists for this service
    const existingNotification = state.notifications.find(
      n => n.type === "health" && 
           (n as HealthNotification).serviceName === serviceName && 
           n.severity === severity && 
           n.title === title &&
           !n.dismissed
    ) as HealthNotification | undefined;
    
    // If similar notification exists and is recent (within 5 minutes), don't add duplicate
    if (existingNotification && (Date.now() - existingNotification.createdAt) < 5 * 60 * 1000) {
      return state;
    }
    
    const newNotification: HealthNotification = {
      ...notificationData,
      id: generateNotificationId("health", serviceName),
      type: "health",
      createdAt: Date.now(),
      read: false,
      dismissed: false,
    };
    
    return { 
      notifications: [...state.notifications, newNotification] 
    };
  }),

  clearServiceNotifications: (serviceName) => set((state) => ({
    notifications: state.notifications.map(notification =>
      notification.type === "health" && 
      (notification as HealthNotification).serviceName === serviceName
        ? { ...notification, dismissed: true, read: true }
        : notification
    )
  })),

  addTaskNotification: (task) => set((state) => {
    // Check for existing notification using both task ID and notification ID
    const taskNotificationId = generateNotificationId("task", task.id);
    const existing = state.notifications.find(
      n => n.id === taskNotificationId || (n.type === "task" && (n as TaskNotification).task.id === task.id)
    );
    if (existing) return state; // Don't duplicate
    
    const taskNotification = taskToNotification(task);
    
    // Save to localStorage for compatibility
    localStorage.setItem(`ingestion-task-${task.id}`, JSON.stringify(task));
    
    return { notifications: [...state.notifications, taskNotification] };
  }),

  updateTaskNotification: (taskId, updates) => set((state) => {
    const notifications = state.notifications.map(notification => {
      if (notification.type === "task" && (notification as TaskNotification).task.id === taskId) {
        const taskNotification = notification as TaskNotification;
        const updatedTask = { ...taskNotification.task, ...updates };
        
        // Clean up localStorage properly to prevent duplicates
        const wasCompleted = taskNotification.task.state !== "PENDING";
        const isNowCompleted = updates.state && updates.state !== "PENDING";
        
        if (isNowCompleted) {
          // Task is completing - remove any pending keys and add completed key
          localStorage.removeItem(`ingestion-task-${taskId}`);
          localStorage.setItem(`completedTask:${taskId}`, JSON.stringify(updatedTask));
        } else if (!isNowCompleted && !wasCompleted) {
          // Task is still pending - update pending key
          localStorage.setItem(`ingestion-task-${taskId}`, JSON.stringify(updatedTask));
        } else {
          // Task state change - clean up old keys and set appropriate new key
          localStorage.removeItem(`ingestion-task-${taskId}`);
          localStorage.removeItem(`completedTask:${taskId}`);
          
          if (isNowCompleted) {
            localStorage.setItem(`completedTask:${taskId}`, JSON.stringify(updatedTask));
          } else {
            localStorage.setItem(`ingestion-task-${taskId}`, JSON.stringify(updatedTask));
          }
        }
        
        // Update notification
        const updatedNotification: TaskNotification = {
          ...taskNotification,
          task: {
            ...updatedTask,
            completedAt: updates.state && updates.state !== "PENDING" ? Date.now() : taskNotification.task.completedAt
          },
          severity: updates.state === "FAILED" ? "error" : 
                   updates.state === "FINISHED" ? "success" : 
                   taskNotification.severity,
          message: updates.state === "FINISHED" 
            ? `Upload completed successfully`
            : updates.state === "FAILED"
            ? `Upload failed`
            : taskNotification.message,
        };
        
        return updatedNotification;
      }
      return notification;
    });
    
    return { notifications };
  }),

  markAsRead: (id) => set((state) => {
    const notifications = state.notifications.map(notification => {
      if (notification.id === id) {
        const updated = { ...notification, read: true };
        
        // Update localStorage for task notifications
        if (notification.type === "task") {
          const taskNotification = notification as TaskNotification;
          const task = { ...taskNotification.task, read: true };
          const key = task.state === "PENDING" ? `ingestion-task-${task.id}` : `completedTask:${task.id}`;
          localStorage.setItem(key, JSON.stringify(task));
        }
        
        return updated;
      }
      return notification;
    });
    
    return { notifications };
  }),

  dismissNotification: (id) => set((state) => ({
    notifications: state.notifications.map(notification =>
      notification.id === id 
        ? { ...notification, dismissed: true, read: true }
        : notification
    )
  })),

  removeNotification: (id) => set((state) => {
    const notification = state.notifications.find(n => n.id === id);
    
    // Remove from localStorage for task notifications
    if (notification?.type === "task") {
      const taskNotification = notification as TaskNotification;
      const taskId = taskNotification.task.id;
      
      // Comprehensive cleanup: remove all possible localStorage keys for this task
      localStorage.removeItem(`ingestion-task-${taskId}`);
      localStorage.removeItem(`completedTask:${taskId}`);
      
      // Also remove any legacy keys that might exist
      Object.keys(localStorage).forEach(key => {
        if (key.includes(taskId) && (key.startsWith('ingestion-task-') || key.startsWith('completedTask:'))) {
          localStorage.removeItem(key);
        }
      });
    }
    
    return { notifications: state.notifications.filter(n => n.id !== id) };
  }),

  clearNotifications: () => set({ notifications: [] }),

  // Clean up duplicate notifications based on task IDs and orphaned localStorage entries
  cleanupDuplicates: () => set((state) => {
    const seenTaskIds = new Set<string>();
    const cleanedNotifications = state.notifications.filter(notification => {
      if (notification.type === "task") {
        const taskNotification = notification as TaskNotification;
        const taskId = taskNotification.task.id;
        
        if (seenTaskIds.has(taskId)) {
          return false; // Filter out duplicate
        }
        seenTaskIds.add(taskId);
      }
      return true;
    });
    
    // Also clean up orphaned localStorage entries that might cause issues
    const currentTaskIds = new Set(
      cleanedNotifications
        .filter(n => n.type === "task")
        .map(n => (n as TaskNotification).task.id)
    );
    
    // Remove localStorage entries for tasks that are no longer in notifications
    Object.keys(localStorage).forEach(key => {
      if (key.startsWith('ingestion-task-') || key.startsWith('completedTask:')) {
        try {
          const task = JSON.parse(localStorage.getItem(key)!);
          if (task?.id && !currentTaskIds.has(task.id)) {
            // This is an orphaned entry - remove it
            localStorage.removeItem(key);
          }
        } catch {
          // Corrupted entry - remove it
          localStorage.removeItem(key);
        }
      }
    });
    
    return { notifications: cleanedNotifications };
  }),

  // Compatibility getters for existing notification system
  getPendingTasks: () => {
    return get().notifications
      .filter((n): n is TaskNotification => n.type === "task" && n.task.state === "PENDING")
      .map(n => notificationToTask(n));
  },

  getCompletedTasks: () => {
    return get().notifications
      .filter((n): n is TaskNotification => n.type === "task" && n.task.state !== "PENDING")
      .map(n => notificationToTask(n));
  },

  getHealthNotifications: () => {
    return get().notifications
      .filter((n): n is HealthNotification => n.type === "health" && !n.dismissed)
      .sort((a, b) => {
        // Sort by severity (error > warning > info), then by newest first
        const severityOrder = { error: 3, warning: 2, info: 1, success: 0 };
        const severityDiff = severityOrder[b.severity] - severityOrder[a.severity];
        if (severityDiff !== 0) return severityDiff;
        
        // Then by unread status
        if (a.read !== b.read) return a.read ? 1 : -1;
        
        // Finally by creation time (newest first)
        return b.createdAt - a.createdAt;
      });
  },

  getAllNotifications: () => {
    return get().notifications
      .filter(n => !n.dismissed)
      .sort((a, b) => {
        // Sort by unread status first (unread items first)
        if (a.read !== b.read) return a.read ? 1 : -1;
        
        // Then by most recent relevant timestamp (newest first)
        // For completed tasks, use completedAt; otherwise use createdAt
        const aTime = a.type === "task" && (a as TaskNotification).task.completedAt 
          ? (a as TaskNotification).task.completedAt!
          : a.createdAt;
        const bTime = b.type === "task" && (b as TaskNotification).task.completedAt
          ? (b as TaskNotification).task.completedAt!
          : b.createdAt;
        
        return bTime - aTime;
      });
  },

  getUnreadCount: () => {
    return get().notifications.filter(n => !n.read && !n.dismissed).length;
  },
}));

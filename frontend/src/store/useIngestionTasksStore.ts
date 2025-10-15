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

/**
 * Extended task type for notifications with completion and read status.
 */
type NotificationTask = IngestionTask & { completedAt?: number; read?: boolean };

/**
 * State interface for the ingestion tasks store.
 */
interface IngestionTasksState {
  // All tasks (pending + completed)
  allTasks: NotificationTask[];
  
  // Hydration (only called once on app start)
  hydrate: () => void;
  
  // Task management
  addTask: (task: IngestionTask) => void;
  updateTask: (taskId: string, updates: Partial<NotificationTask>) => void;
  markAsRead: (taskId: string) => void;
  removeTask: (taskId: string) => void;
  
  // Getters (as functions, not properties)
  getPendingTasks: () => IngestionTask[];
  getCompletedTasks: () => NotificationTask[];
  getUnreadCount: () => number;
  
  // Legacy properties for compatibility
  pendingTasks: IngestionTask[];
  completedTasks: NotificationTask[];
  unreadCount: number;
}

/**
 * Zustand store for managing ingestion task notifications and status tracking.
 * 
 * Manages all ingestion tasks including pending and completed tasks, tracks read/unread
 * status for notifications, provides filtering and counting capabilities. Handles
 * task lifecycle from creation through completion with notification management.
 * 
 * @returns Store with task state and management functions
 * 
 * @example
 * ```tsx
 * const { getPendingTasks, getUnreadCount, addTask, markAsRead } = useIngestionTasksStore();
 * const pending = getPendingTasks();
 * markAsRead(taskId);
 * ```
 */
export const useIngestionTasksStore = create<IngestionTasksState>((set, get) => ({
  allTasks: [],

  hydrate: () => {
    const tasks: NotificationTask[] = [];
    
    // Load from localStorage once
    Object.keys(localStorage).forEach(key => {
      if (key.startsWith('completedTask:') || key.startsWith('ingestion-task-')) {
        try {
          const task = JSON.parse(localStorage.getItem(key)!);
          tasks.push(task);
        } catch {
          localStorage.removeItem(key); // Clean up corrupted entries
        }
      }
    });

    set({ allTasks: tasks });
  },

  addTask: (task) => set((state) => {
    const existing = state.allTasks.find(t => t.id === task.id);
    if (existing) return state; // Don't duplicate
    
    const newTask: NotificationTask = { ...task, read: false };
    
    // Save to localStorage
    localStorage.setItem(`ingestion-task-${task.id}`, JSON.stringify(newTask));
    
    return { allTasks: [...state.allTasks, newTask] };
  }),

  updateTask: (taskId, updates) => set((state) => {
    const allTasks = state.allTasks.map(task => {
      if (task.id === taskId) {
        const updated = { ...task, ...updates };
        
        // If task completed, move to completed storage
        if (updates.state && updates.state !== "PENDING") {
          localStorage.removeItem(`ingestion-task-${taskId}`);
          localStorage.setItem(`completedTask:${taskId}`, JSON.stringify(updated));
        } else {
          localStorage.setItem(`ingestion-task-${taskId}`, JSON.stringify(updated));
        }
        
        return updated;
      }
      return task;
    });
    
    return { allTasks };
  }),

  markAsRead: (taskId) => set((state) => {
    const allTasks = state.allTasks.map(task => {
      if (task.id === taskId) {
        const updated = { ...task, read: true };
        
        // Update localStorage
        const key = task.state === "PENDING" ? `ingestion-task-${taskId}` : `completedTask:${taskId}`;
        localStorage.setItem(key, JSON.stringify(updated));
        
        return updated;
      }
      return task;
    });
    
    return { allTasks };
  }),

  removeTask: (taskId) => set((state) => {
    // Remove from localStorage
    localStorage.removeItem(`ingestion-task-${taskId}`);
    localStorage.removeItem(`completedTask:${taskId}`);
    
    return { allTasks: state.allTasks.filter(task => task.id !== taskId) };
  }),

  // Function getters
  getPendingTasks: () => {
    return get().allTasks.filter(task => task.state === "PENDING");
  },

  getCompletedTasks: () => {
    return get().allTasks.filter(task => task.state !== "PENDING");
  },

  getUnreadCount: () => {
    return get().allTasks.filter(task => !task.read).length;
  },

  // Legacy computed properties for compatibility
  get pendingTasks() {
    return get().allTasks.filter(task => task.state === "PENDING");
  },

  get completedTasks() {
    return get().allTasks.filter(task => task.state !== "PENDING");
  },

  get unreadCount() {
    return get().allTasks.filter(task => !task.read).length;
  },
}));

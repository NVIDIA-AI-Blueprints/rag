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

import { useEffect, useState, useRef } from "react";
import { Popover, Button } from "@kui/react";
import { useNotificationStore } from "../../store/useNotificationStore";
import { NotificationDropdown } from "./NotificationDropdown";
import { NotificationBadge } from "./NotificationBadge";
import { TaskPoller } from "./TaskPoller";
import type { TaskNotification } from "../../types/notifications";

/**
 * Global reference to notification panel control functions for external access.
 */
let globalNotificationOpen: (() => void) | null = null;
let globalNotificationToggle: (() => void) | null = null;

/**
 * Global function to open the notification panel from anywhere in the application.
 * Unlike toggle, this will always open it (no effect if already open).
 * 
 * @example
 * ```tsx
 * import { openNotificationPanel } from './NotificationBell';
 * openNotificationPanel(); // Opens the notification dropdown
 * ```
 */
/* eslint-disable-next-line react-refresh/only-export-components */
export const openNotificationPanel = () => {
  if (globalNotificationOpen) {
    globalNotificationOpen();
  }
};

/**
 * Global function to toggle the notification panel from anywhere in the application.
 */
/* eslint-disable-next-line react-refresh/only-export-components */
export const toggleNotificationPanel = () => {
  if (globalNotificationToggle) {
    globalNotificationToggle();
  }
};

/**
 * Notification bell component that displays task notifications and manages popover state.
 * 
 * Shows a bell icon with a badge indicating unread notifications. When clicked, opens
 * a KUI Popover showing pending and completed ingestion tasks. Automatically polls for
 * task updates and manages global notification panel access.
 * 
 * @returns Notification bell component with popover functionality using KUI components
 */
export default function NotificationBell() {
  const [isOpen, setIsOpen] = useState(false);
  const hasHydrated = useRef(false);
  const { 
    notifications,
    hydrate,
    cleanupDuplicates 
  } = useNotificationStore();

  // Hydrate from localStorage on mount (only once) and cleanup duplicates
  useEffect(() => {
    // Only hydrate once, regardless of notifications.length changes
    if (!hasHydrated.current) {
      hasHydrated.current = true;
      hydrate();
      
      // Clean up any existing duplicates after initial hydration
      const timeoutId = setTimeout(() => cleanupDuplicates(), 100);
      
      // Return cleanup function to clear timeout
      return () => {
        clearTimeout(timeoutId);
      };
    }
  }, [hydrate, cleanupDuplicates]);

  // Set global references for external access (open and toggle)
  useEffect(() => {
    globalNotificationOpen = () => setIsOpen(true);
    globalNotificationToggle = () => setIsOpen(prev => !prev);
    return () => {
      globalNotificationOpen = null;
      globalNotificationToggle = null;
    };
  }, []);

  // Calculate unread count reactively from notifications
  const unreadCount = notifications.filter(n => !n.read && !n.dismissed).length;
  
  // Get pending tasks for TaskPoller components
  const pendingTasks = notifications
    .filter((n): n is TaskNotification => n.type === "task" && !n.dismissed && n.task.state === "PENDING")
    .map(n => n.task);

  return (
    <>
      <Popover
        open={isOpen}
        onOpenChange={setIsOpen}
        side="bottom"
        align="end"
        slotContent={<NotificationDropdown />}
      >
        <Button 
          kind="tertiary" 
          size="small"
        >
          <NotificationBadge count={unreadCount} />
        </Button>
      </Popover>

      {pendingTasks.map((task) => (
        <TaskPoller key={task.id} taskId={task.id} />
      ))}
    </>
  );
}

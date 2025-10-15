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
import { Popover, Button, ThemeProvider } from "@kui/react";
import { useNotificationStore } from "../../store/useNotificationStore";
import { NotificationDropdown } from "./NotificationDropdown";
import { NotificationBadge } from "./NotificationBadge";
import { TaskPoller } from "./TaskPoller";
import type { TaskNotification } from "../../types/notifications";

/**
 * Global reference to notification toggle function for external access.
 */
let globalNotificationToggle: (() => void) | null = null;

/**
 * Global function to open the notification panel from anywhere in the application.
 * 
 * @example
 * ```tsx
 * import { openNotificationPanel } from './NotificationBell';
 * openNotificationPanel(); // Opens the notification dropdown
 * ```
 */
/* eslint-disable-next-line react-refresh/only-export-components */
export const openNotificationPanel = () => {
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

  // Set global reference for external access
  useEffect(() => {
    globalNotificationToggle = () => setIsOpen(true);
    return () => {
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
    <ThemeProvider theme="dark">
      <Popover
        open={isOpen}
        onOpenChange={setIsOpen}
        side="bottom"
        align="end"
        slotContent={
          <NotificationDropdown />
        }
        style={{ background: 'var(--background-color-interaction-inverse)' }}
      >
        <Button 
          kind="tertiary" 
          size="small"
          className="relative"
        >
          <NotificationBadge count={unreadCount} />
        </Button>
      </Popover>

      {pendingTasks.map((task) => (
        <TaskPoller key={task.id} taskId={task.id} />
      ))}
    </ThemeProvider>
  );
}

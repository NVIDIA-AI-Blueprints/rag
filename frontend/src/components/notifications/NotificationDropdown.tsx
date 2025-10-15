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

import { useCallback, useMemo } from "react";
import { Stack, Text, Divider } from "@kui/react";
import type { HealthNotification, TaskNotification } from "../../types/notifications";
import { TaskDisplay } from "../tasks/TaskDisplay";
import { HealthNotificationDisplay } from "./HealthNotificationDisplay";
import { useNotificationStore } from "../../store/useNotificationStore";

export const NotificationDropdown = () => {
  const { 
    getAllNotifications, 
    markAsRead, 
    removeNotification 
  } = useNotificationStore();

  // Get all notifications from the unified store
  const allNotifications = getAllNotifications();

  const handleMarkAsRead = useCallback((id: string) => {
    markAsRead(id);
  }, [markAsRead]);

  const handleRemove = useCallback((id: string) => {
    removeNotification(id);
  }, [removeNotification]);

  // Group notifications by type
  const groupedNotifications = useMemo(() => {
    const health = allNotifications.filter((n): n is HealthNotification => n.type === "health");
    const tasks = allNotifications.filter((n): n is TaskNotification => n.type === "task");
    
    return { health, tasks };
  }, [allNotifications]);

  const totalNotifications = allNotifications.length;

  if (totalNotifications === 0) {
    return (
      <div style={{ width: '600px', maxHeight: '384px', padding: '16px' }}>
        <Text kind="body/regular/sm" className="text-subtle">
          No notifications
        </Text>
      </div>
    );
  }

  return (
    <Stack 
      gap="3" 
      style={{ 
        width: '50vw', 
        maxHeight: '484px', 
        overflowY: 'auto', 
        background: 'var(--background-color-interaction-inverse)',
        padding: '8px'
      }}
    >
      {/* Health Notifications Section */}
      {groupedNotifications.health.length > 0 && (
        <>
          <Text kind="body/semibold/sm" className="text-white px-2">
            Service Health Issues ({groupedNotifications.health.length})
          </Text>
          <Stack gap="2">
            {groupedNotifications.health.map((notification) => (
              <HealthNotificationDisplay
                key={notification.id}
                notification={notification}
                onMarkRead={() => handleMarkAsRead(notification.id)}
                onRemove={() => handleRemove(notification.id)}
              />
            ))}
          </Stack>
        </>
      )}

      {/* Divider between sections if both exist */}
      {groupedNotifications.health.length > 0 && groupedNotifications.tasks.length > 0 && (
        <Divider />
      )}

      {/* Task Notifications Section */}
      {groupedNotifications.tasks.length > 0 && (
        <>
          <Text kind="body/semibold/sm" className="text-white px-2">
            Ingestion Tasks ({groupedNotifications.tasks.length})
          </Text>
          <Stack gap="2">
            {groupedNotifications.tasks.map((notification) => (
              <TaskDisplay 
                key={notification.id}
                task={{
                  ...notification.task,
                  read: notification.read,
                }}
                onMarkRead={() => handleMarkAsRead(notification.id)}
                onRemove={() => handleRemove(notification.id)}
              />
            ))}
          </Stack>
        </>
      )}
    </Stack>
  );
}; 
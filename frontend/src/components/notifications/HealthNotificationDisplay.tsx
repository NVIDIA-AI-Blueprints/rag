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

import { useCallback } from "react";
import { Card, Button, Text, Stack, Flex } from "@kui/react";
import type { HealthNotification, NotificationSeverity } from "../../types/notifications";

interface HealthNotificationDisplayProps {
  notification: HealthNotification;
  onMarkRead?: () => void;
  onRemove?: () => void;
}

/**
 * Get severity icon and styling for health notifications.
 */
const getSeverityDisplay = (severity: NotificationSeverity) => {
  switch (severity) {
    case "error":
      return {
        icon: "🔴",
        color: "text-red-400",
        bgColor: "bg-red-900/20",
      };
    case "warning":
      return {
        icon: "🟡",
        color: "text-yellow-400",
        bgColor: "bg-yellow-900/20",
      };
    case "info":
      return {
        icon: "ℹ️",
        color: "text-blue-400",
        bgColor: "bg-blue-900/20",
      };
    case "success":
      return {
        icon: "✅",
        color: "text-green-400",
        bgColor: "bg-green-900/20",
      };
  }
};

/**
 * Format relative time for notification display.
 */
const formatRelativeTime = (timestamp: number): string => {
  const now = Date.now();
  const diff = now - timestamp;
  const minutes = Math.floor(diff / (1000 * 60));
  const hours = Math.floor(diff / (1000 * 60 * 60));
  const days = Math.floor(diff / (1000 * 60 * 60 * 24));

  if (minutes < 1) return "Just now";
  if (minutes < 60) return `${minutes}m ago`;
  if (hours < 24) return `${hours}h ago`;
  return `${days}d ago`;
};

/**
 * Remove icon component.
 */
const RemoveIcon = () => (
  <svg 
    className="w-4 h-4" 
    fill="none" 
    stroke="currentColor" 
    strokeWidth="2" 
    viewBox="0 0 24 24"
    data-testid="remove-icon"
  >
    <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
  </svg>
);

/**
 * Health notification display component that shows health service issues.
 * 
 * Displays health notification with severity indicator, service information,
 * and action buttons. Follows the same design patterns as TaskDisplay for
 * consistency within the notification system.
 * 
 * @param props - Component props
 * @returns Health notification display component
 */
export function HealthNotificationDisplay({ 
  notification, 
  onMarkRead, 
  onRemove 
}: HealthNotificationDisplayProps) {
  const { icon, color } = getSeverityDisplay(notification.severity);
  
  const handleMarkAsRead = useCallback(() => {
    if (!notification.read && onMarkRead) {
      onMarkRead();
    }
  }, [notification.read, onMarkRead]);

  const clickableProps = onMarkRead && !notification.read
    ? {
        tabIndex: 0,
        onClick: handleMarkAsRead,
        onFocus: handleMarkAsRead,
        onKeyDown: (e: React.KeyboardEvent) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            handleMarkAsRead();
          }
        },
      }
    : {};

  return (
    <div data-testid="health-notification-display" {...clickableProps}>
      <Card
        interactive={onMarkRead && !notification.read}
        kind={notification.read ? "solid" : "float"}
        selected={!notification.read}
        style={{ background: 'var(--border-color-interaction-inverse-pressed)' }}
        className={`group relative transition-all duration-200 ${
          notification.read 
            ? 'opacity-75' 
            : 'shadow-lg shadow-[var(--nv-green)]/20'
        } ${onMarkRead && !notification.read ? 'cursor-pointer' : ''}`}
      >
        {/* Remove button */}
        {onRemove && (
          <Button
            kind="tertiary"
            size="small"
            color="danger"
            onClick={(e) => {
              e.stopPropagation();
              onRemove();
            }}
            title="Remove notification"
            className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity z-10"
            data-testid="remove-button"
          >
            <RemoveIcon />
          </Button>
        )}

        <Stack gap="3">
          {/* Header */}
          <Flex gap="3" align="start" data-testid="health-notification-header">
            <div className="flex items-center mt-0.5">
              <Text style={{ fontSize: '16px' }} data-testid="severity-icon">
                {icon}
              </Text>
            </div>
            
            <Stack gap="1" style={{ flex: 1 }}>
              <Text 
                kind="body/semibold/md"
                className={notification.read ? 'text-neutral-400' : 'text-white'}
                data-testid="notification-title"
              >
                {notification.title}
              </Text>
              <Text 
                kind="body/regular/xs"
                className={color}
                data-testid="service-info"
              >
                {notification.serviceName} • {formatRelativeTime(notification.createdAt)}
              </Text>
            </Stack>
          </Flex>

          {/* Message */}
          {notification.message && (
            <Stack gap="2" data-testid="notification-content">
              <Text 
                kind="body/regular/sm"
                className={notification.read ? 'text-neutral-500' : 'text-gray-300'}
                style={{ lineHeight: '1.4' }}
                data-testid="notification-message"
              >
                {notification.message}
              </Text>
              
              {/* Additional service details */}
              {notification.url && (
                <Text 
                  kind="body/regular/xs"
                  className="text-neutral-500"
                  data-testid="service-url"
                >
                  Service URL: {notification.url}
                </Text>
              )}
              
              {notification.latency && (
                <Text 
                  kind="body/regular/xs"
                  className="text-neutral-500"
                  data-testid="service-latency"
                >
                  Response time: {notification.latency}ms
                </Text>
              )}
              
              {/* Category context */}
              <Text 
                kind="body/regular/xs"
                className="text-neutral-600"
                data-testid="service-category"
              >
                Category: {notification.serviceCategory}
              </Text>
            </Stack>
          )}
        </Stack>
      </Card>
    </div>
  );
}

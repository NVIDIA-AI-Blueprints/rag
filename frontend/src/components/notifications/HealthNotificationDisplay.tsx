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
import { X, AlertCircle, AlertTriangle, Info, CheckCircle } from "lucide-react";
import type { HealthNotification, NotificationSeverity } from "../../types/notifications";

interface HealthNotificationDisplayProps {
  notification: HealthNotification;
  onMarkRead?: () => void;
  onRemove?: () => void;
}

/**
 * Get severity icon for health notifications.
 */
const getSeverityIcon = (severity: NotificationSeverity) => {
  switch (severity) {
    case "error":
      return <AlertCircle size={16} />;
    case "warning":
      return <AlertTriangle size={16} />;
    case "info":
      return <Info size={16} />;
    case "success":
      return <CheckCircle size={16} />;
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
        kind="solid"
        selected={!notification.read}
        className={onMarkRead && !notification.read ? 'cursor-pointer' : ''}
      >
        <Stack gap="3">
          {/* Header */}
          <Flex gap="density-md" align="start" justify="between" data-testid="health-notification-header">
            <Flex gap="density-md" align="start">
              <div style={{ marginTop: '2px' }} data-testid="severity-icon">
                {getSeverityIcon(notification.severity)}
              </div>
              
              <Stack gap="1">
                <Text 
                  kind="body/semibold/md"
                  data-testid="notification-title"
                >
                  {notification.title}
                </Text>
                <Text 
                  kind="body/regular/xs"
                  data-testid="service-info"
                >
                  {notification.serviceName} • {formatRelativeTime(notification.createdAt)}
                </Text>
              </Stack>
            </Flex>

            {/* Remove button */}
            {onRemove && (
              <Button
                kind="tertiary"
                size="tiny"
                onClick={(e) => {
                  e.stopPropagation();
                  onRemove();
                }}
                title="Remove notification"
                data-testid="remove-button"
              >
                <X size={16} />
              </Button>
            )}
          </Flex>

          {/* Message */}
          {notification.message && (
            <Stack gap="2" data-testid="notification-content">
              <Text 
                kind="body/regular/sm"
                data-testid="notification-message"
              >
                {notification.message}
              </Text>
              
              {/* Additional service details */}
              {notification.url && (
                <Text 
                  kind="body/regular/xs"
                  data-testid="service-url"
                >
                  Service URL: {notification.url}
                </Text>
              )}
              
              {notification.latency && (
                <Text 
                  kind="body/regular/xs"
                  data-testid="service-latency"
                >
                  Response time: {notification.latency}ms
                </Text>
              )}
              
              {/* Category context */}
              <Text 
                kind="body/regular/xs"
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

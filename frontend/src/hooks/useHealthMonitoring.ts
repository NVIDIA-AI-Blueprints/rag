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
import { useHealthStatus } from "../api/useHealthApi";
import { useNotificationStore } from "../store/useNotificationStore";
import type { HealthResponse } from "../types/api";
import type { NotificationSeverity, HealthNotification } from "../types/notifications";

/**
 * Service health info interface for processing.
 */
interface ServiceHealthInfo {
  service: string;
  status: string;
  error?: string | null;
  url?: string;
  latency_ms?: number;
  message?: string;
}

/**
 * Map service status to notification severity.
 */
const getNotificationSeverity = (status: string): NotificationSeverity => {
  switch (status.toLowerCase()) {
    case "error":
    case "unhealthy":
      return "error";
    case "timeout":
    case "skipped":
      return "warning";
    default:
      return "info";
  }
};

/**
 * Generate user-friendly notification title and message based on service and status.
 */
const generateNotificationContent = (service: ServiceHealthInfo, category: string) => {
  const serviceName = service.service;
  const status = service.status.toLowerCase();
  
  let title: string;
  let message: string;
  
  switch (status) {
    case "error":
    case "unhealthy":
      title = `${serviceName} Service Error`;
      message = service.error 
        ? `${serviceName} is experiencing issues: ${service.error}`
        : `${serviceName} service is not responding properly. Please check the service configuration.`;
      break;
      
    case "timeout":
      title = `${serviceName} Connection Timeout`;
      message = `${serviceName} service is not responding within the expected time limit. This may indicate network issues or service overload.`;
      break;
      
    case "skipped":
      title = `${serviceName} Service Skipped`;
      message = service.error
        ? `${serviceName} health check was skipped: ${service.error}`
        : `${serviceName} service health check was skipped due to configuration.`;
      break;
      
    default:
      title = `${serviceName} Service Issue`;
      message = `${serviceName} service status: ${service.status}`;
  }
  
  // Add category context for better understanding
  const categoryContext = getCategoryContext(category);
  if (categoryContext) {
    message += ` ${categoryContext}`;
  }
  
  return { title, message };
};

/**
 * Get contextual information about service categories.
 */
const getCategoryContext = (category: string): string => {
  switch (category) {
    case "databases":
      return "This may affect document storage and retrieval functionality.";
    case "nim":
      return "This may affect AI model inference and chat functionality.";
    case "processing":
      return "This may affect document ingestion and processing.";
    case "task_management":
      return "This may affect background task processing.";
    case "object_storage":
      return "This may affect file storage and retrieval.";
    default:
      return "";
  }
};

/**
 * Process health response and create notifications for unhealthy services.
 */
const processHealthData = (
  health: HealthResponse,
  addHealthNotification: (notification: HealthNotification) => void,
  clearServiceNotifications: (serviceName: string) => void
) => {
  const categories = ['databases', 'nim', 'processing', 'task_management', 'object_storage'] as const;
  
  categories.forEach(category => {
    const services = health[category] as ServiceHealthInfo[] | undefined;
    
    if (!services || !Array.isArray(services)) return;
    
    services.forEach(service => {
      const isUnhealthy = ['error', 'unhealthy', 'timeout'].includes(service.status.toLowerCase());
      
      if (isUnhealthy) {
        const { title, message } = generateNotificationContent(service, category);
        const severity = getNotificationSeverity(service.status);
        
        addHealthNotification({
          id: `health-${service.service}-${Date.now()}`,
          type: "health" as const,
          title,
          message,
          severity,
          serviceName: service.service,
          serviceCategory: category,
          url: service.url,
          latency: service.latency_ms,
          createdAt: Date.now(),
          read: false,
          dismissed: false,
        });
      } else if (service.status.toLowerCase() === 'healthy') {
        // Clear any existing notifications for this service since it's now healthy
        clearServiceNotifications(service.service);
      }
    });
  });
};

/**
 * Hook to monitor service health and trigger notifications for unhealthy services.
 * 
 * Automatically monitors the health status endpoint and creates notifications
 * when services become unhealthy. Clears notifications when services recover.
 * Uses React Query for efficient polling and caching.
 * 
 * @returns Health monitoring status
 * 
 * @example
 * ```tsx
 * function App() {
 *   useHealthMonitoring(); // Automatically monitors health in background
 *   
 *   return <div>App content</div>;
 * }
 * ```
 */
export function useHealthMonitoring() {
  const { data: health, isLoading, error, isError } = useHealthStatus();
  const { addHealthNotification, clearServiceNotifications } = useNotificationStore();
  
  // Track previous health data to avoid duplicate notifications
  const previousHealthRef = useRef<HealthResponse | null>(null);
  
  useEffect(() => {
    // Only process health data if it's different from the previous check
    if (health && !isLoading && JSON.stringify(health) !== JSON.stringify(previousHealthRef.current)) {
      processHealthData(health, addHealthNotification, clearServiceNotifications);
      previousHealthRef.current = health;
    }
  }, [health, isLoading, addHealthNotification, clearServiceNotifications]);
  
  // Handle query errors
  useEffect(() => {
    if (isError && error) {
      addHealthNotification({
        title: "Health Check Failed",
        message: `Unable to check service health: ${error.message}. This may indicate connectivity issues with the backend services.`,
        severity: "error",
        serviceName: "Health Monitor",
        serviceCategory: "system",
      });
    }
  }, [isError, error, addHealthNotification]);
  
  return {
    isLoading,
    isError,
    hasHealthData: !!health,
  };
}

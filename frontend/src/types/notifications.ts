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

import type { IngestionTask } from "./api";

/**
 * Notification severity levels.
 */
export type NotificationSeverity = "error" | "warning" | "info" | "success";

/**
 * Base notification interface.
 */
export interface BaseNotification {
  id: string;
  type: "health" | "task";
  title: string;
  message?: string;
  severity: NotificationSeverity;
  createdAt: number;
  read: boolean;
  dismissed: boolean;
}

/**
 * Health notification specific to service health issues.
 */
export interface HealthNotification extends BaseNotification {
  type: "health";
  serviceName: string;
  serviceCategory: string; // 'databases', 'nim', 'processing', etc.
  url?: string;
  latency?: number;
}

/**
 * Task notification for ingestion tasks.
 */
export interface TaskNotification extends BaseNotification {
  type: "task";
  task: IngestionTask & { completedAt?: number };
  collectionName: string;
}

/**
 * Union type for all notification types.
 */
export type AppNotification = HealthNotification | TaskNotification;

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

import { Badge } from "@kui/react";

/**
 * Props for the NotificationBadge component.
 */
interface NotificationBadgeProps {
  count: number;
}

/**
 * Bell icon SVG component.
 */
const BellIcon = () => (
  <svg
    className="w-5 h-5"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    viewBox="0 0 24 24"
  >
    <path strokeLinecap="round" strokeLinejoin="round" d="M14.857 17.082a23.848 23.848 0 005.454-1.31A8.967 8.967 0 0118 9.75v-.7V9A6 6 0 006 9v.75a8.967 8.967 0 01-2.312 6.022c1.733.64 3.56 1.085 5.455 1.31m5.714 0a24.255 24.255 0 01-5.714 0m5.714 0a3 3 0 11-5.714 0" />
  </svg>
);

/**
 * Notification badge component that displays a bell icon with optional count badge.
 * 
 * Shows a bell icon and conditionally displays a KUI Badge with the notification
 * count when there are unread notifications.
 * 
 * @param props - Component props with notification count
 * @returns Bell icon with optional notification count badge using KUI Badge
 */
export const NotificationBadge = ({ count }: NotificationBadgeProps) => (
  <div className="relative">
    <BellIcon />
    {count > 0 && (
      <div 
        className="absolute" 
        style={{ 
          top: '-4px', 
          right: '-24px'
        }}
      >
        <Badge kind="solid" color="green">
          {count}
        </Badge>
      </div>
    )}
  </div>
); 
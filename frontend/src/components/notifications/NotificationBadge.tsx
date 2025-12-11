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
import { Bell } from "lucide-react";

/**
 * Props for the NotificationBadge component.
 */
interface NotificationBadgeProps {
  count: number;
}

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
    <Bell size={20} />
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

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

import { StatusMessage, Flex } from "@kui/react";
import { FileText } from "lucide-react";

/**
 * Props for the EmptyState component.
 */
interface EmptyStateProps {
  title: string;
  description?: string;
  icon?: React.ReactNode;
}

/**
 * Empty state component for displaying placeholder content when no data is available.
 */
export const EmptyState = ({ 
  title, 
  description, 
  icon = <FileText size={40} style={{ color: 'var(--text-color-subtle)' }} />
}: EmptyStateProps) => (
  <Flex 
    direction="col" 
    align="center" 
    justify="center" 
    style={{ height: '300px', textAlign: 'center', padding: 'var(--spacing-density-md)' }}
  >
    <StatusMessage
      slotHeading={title}
      slotSubheading={description}
      slotMedia={icon}
    />
  </Flex>
);

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

import React from 'react';
import { VerticalNav, Text, Stack } from "@kui/react";
import { BarChart3, Sliders, Cpu, Globe, Settings } from "lucide-react";

interface SettingsSidebarProps {
  activeSection: string;
  onSectionChange: (section: string) => void;
}

const settingsNavItems = [
  { id: 'ragConfig', title: 'RAG Configuration', icon: <BarChart3 size={20} /> },
  { id: 'features', title: 'Feature Toggles', icon: <Sliders size={20} /> },
  { id: 'models', title: 'Model Configuration', icon: <Cpu size={20} /> },
  { id: 'endpoints', title: 'Endpoint Configuration', icon: <Globe size={20} /> },
  { id: 'advanced', title: 'Other Settings', icon: <Settings size={20} /> },
];

export const SettingsSidebar: React.FC<SettingsSidebarProps> = ({
  activeSection,
  onSectionChange,
}) => {
  return (
    <div style={{ 
      width: '280px',
      borderRight: '1px solid var(--border-color-base)',
      padding: 'var(--spacing-density-lg)'
    }}>
      <Stack gap="density-sm">
        <Text 
          kind="label/bold/sm" 
          style={{ 
            color: 'var(--text-color-subtle)', 
            textTransform: 'uppercase',
            letterSpacing: '0.05em',
            marginBottom: 'var(--spacing-density-md)'
          }}
        >
          Configuration
        </Text>
        
        <VerticalNav
          items={settingsNavItems.map((item) => ({
            id: item.id,
            slotLabel: item.title,
            slotIcon: item.icon,
            active: activeSection === item.id,
            href: `#${item.id}`,
            attributes: {
              VerticalNavLink: {
                onClick: (e: React.MouseEvent) => {
                  e.preventDefault();
                  onSectionChange(item.id);
                }
              }
            }
          }))}
        />
      </Stack>
    </div>
  );
};

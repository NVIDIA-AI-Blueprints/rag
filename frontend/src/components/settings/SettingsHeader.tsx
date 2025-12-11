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

import { Flex, Text, Divider } from "@kui/react";
import { Settings } from "lucide-react";

/**
 * Header component for the settings page displaying title and description.
 */
export const SettingsHeader = () => (
  <div data-testid="settings-header">
    <Flex align="center" gap="density-md" style={{ paddingBottom: 'var(--spacing-density-lg)' }}>
      <div 
        style={{ 
          padding: 'var(--spacing-density-sm)',
          borderRadius: 'var(--border-radius-xl)',
          backgroundColor: 'var(--color-brand-alpha-15)',
          border: '1px solid var(--color-brand-alpha-25)'
        }}
        data-testid="settings-icon-container"
      >
        <Settings 
          size={24} 
          style={{ color: 'var(--color-brand)' }}
          data-testid="settings-icon"
        />
      </div>
      <div data-testid="settings-content">
        <Text 
          kind="title/lg"
          data-testid="settings-title"
        >
          Settings
        </Text>
        <Text 
          kind="body/regular/sm"
          style={{ color: 'var(--text-color-subtle)' }}
          data-testid="settings-description"
        >
          Configure your RAG system parameters and preferences
        </Text>
      </div>
    </Flex>
    <Divider />
  </div>
);

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
import { Panel, Stack, Text } from "@kui/react";
import { RagConfigSection } from './RagConfigSection';
import { FeatureTogglesSection } from './FeatureTogglesSection';
import { ModelsSection } from './ModelsSection';
import { EndpointsSection } from './EndpointsSection';
import { AdvancedSection } from './AdvancedSection';
import { 
  RagIcon, 
  FeaturesIcon, 
  ModelsIcon, 
  EndpointsIcon, 
  AdvancedIcon 
} from '../icons';

interface SettingsContentProps {
  activeSection: string;
  onShowWarning?: (featureKey: string, newValue: boolean) => void;
}

export const SettingsContent: React.FC<SettingsContentProps> = ({
  activeSection,
  onShowWarning,
}) => {
  const renderContent = () => {
    switch (activeSection) {
      case 'ragConfig':
        return (
          <Panel
            slotHeading="RAG Configuration"
            slotIcon={<RagIcon className="w-6 h-6" />}
            elevation="mid"
            density="standard"
          >
            <Stack gap="4">
              <Text kind="body/regular/sm">
                Configure retrieval-augmented generation settings and parameters.
              </Text>
              <RagConfigSection />
            </Stack>
          </Panel>
        );
      
      case 'features':
        return (
          <Panel
            slotHeading="Feature Toggles"
            slotIcon={<FeaturesIcon className="w-6 h-6" />}
            elevation="mid"
            density="standard"
          >
            <Stack gap="4">
              <Text kind="body/regular/sm">
                Manage feature flags and experimental functionality.
              </Text>
              <FeatureTogglesSection onShowWarning={onShowWarning || (() => {})} />
            </Stack>
          </Panel>
        );

      case 'models':
        return (
          <Panel
            slotHeading="Model Configuration"
            slotIcon={<ModelsIcon className="w-6 h-6" />}
            elevation="mid"
            density="standard"
          >
              <Stack gap="4">
                <Text kind="body/regular/sm">
                  Configure AI models for language generation, embeddings, and reranking.
              </Text>
              <ModelsSection />
            </Stack>
          </Panel>
        );

      case 'endpoints':
        return (
          <Panel
            slotHeading="Endpoint Configuration"
            slotIcon={<EndpointsIcon className="w-6 h-6" />}
            elevation="mid"
            density="standard"
          >
              <Stack gap="4">
                <Text kind="body/regular/sm">
                  Set up API endpoints and connection settings.
              </Text>
              <EndpointsSection />
            </Stack>
          </Panel>
        );

      case 'advanced':
        return (
          <Panel
            slotHeading="Advanced Settings"
            slotIcon={<AdvancedIcon className="w-6 h-6" />}
            elevation="mid"
            density="standard"
          >
              <Stack gap="4">
                <Text kind="body/regular/sm">
                  Advanced configuration options and system settings.
              </Text>
              <AdvancedSection />
            </Stack>
          </Panel>
        );

      default:
        return (
          <Panel
            slotHeading="RAG Configuration"
            slotIcon={<RagIcon className="w-6 h-6" />}
            elevation="mid"
            density="standard"
          >
              <Stack gap="4">
                <Text kind="body/regular/sm">
                  Configure retrieval-augmented generation settings and parameters.
              </Text>
              <RagConfigSection />
            </Stack>
          </Panel>
        );
    }
  };

  return (
    <Stack gap="6" padding="6">
      {renderContent()}
    </Stack>
  );
}; 